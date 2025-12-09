from graphframes import GraphFrame
from pyspark.sql import SparkSession
import trimesh
from pyspark.sql.functions import floor, col, row_number, desc
from pyspark.sql.window import Window
import numpy as np
import os


def setup_environment():
    """Java 및 PySpark 환경 변수 설정"""
    os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
    graphframes_package = "graphframes:graphframes:0.8.3-spark3.5-s_2.12"
    os.environ["PYSPARK_SUBMIT_ARGS"] = f"--packages {graphframes_package} pyspark-shell"


def create_spark_session():
    """SparkSession 생성"""
    spark = SparkSession.builder \
        .appName("GraphFramesExample") \
        .getOrCreate()
    print(f"Spark Version: {spark.version}")
    return spark


def load_mesh(mesh_path):
    """메쉬 파일 로드"""
    mesh = trimesh.load(mesh_path)
    return mesh


def create_graph_dataframes(spark, mesh):
    """정점 및 간선 DataFrame 생성"""
    vertices_df = spark.createDataFrame(
        [(i, float(v[0]), float(v[1]), float(v[2])) 
         for i, v in enumerate(mesh.vertices)],
        ["id", "x", "y", "z"]
    )
    
    edges_data = []
    for face in mesh.faces:
        v0, v1, v2 = face
        edges_data.extend([(int(v0), int(v1)), (int(v1), int(v2)), (int(v2), int(v0))])
    
    edges_df = spark.createDataFrame(edges_data, ["src", "dst"]).distinct()
    
    return vertices_df, edges_df


def calculate_triangle_counts(vertices_df, edges_df):
    """GraphFrame을 사용하여 삼각형 개수 계산"""
    g = GraphFrame(vertices_df, edges_df)
    triangle_counts = g.triangleCount()
    return triangle_counts


def calculate_voxel_size(mesh, dim=20):
    """복셀 크기 계산"""
    bounding_box_diagonal = np.linalg.norm(mesh.extents)
    voxel_size = bounding_box_diagonal / dim
    print(f"복셀 크기(Voxel Size): {voxel_size:.4f}")
    return voxel_size


def cluster_vertices(triangle_counts, vertices_df, voxel_size):
    """정점 클러스터링 및 대표 정점 선정"""
    # 정점 데이터와 삼각형 개수 데이터 결합
    # triangle_counts에서 count 컬럼만 선택하여 조인
    vertex_info_df = vertices_df.join(
        triangle_counts.select("id", "count"), 
        "id"
    )
    
    # 각 정점의 복셀 좌표 계산
    clustered_df = vertex_info_df.withColumn("vx", floor(col("x") / voxel_size)) \
                                 .withColumn("vy", floor(col("y") / voxel_size)) \
                                 .withColumn("vz", floor(col("z") / voxel_size))
    
    # 각 복셀 내에서 삼각형 개수가 가장 많은 정점을 대표 정점으로 선정
    windowSpec = Window.partitionBy("vx", "vy", "vz").orderBy(desc("count"))
    ranked_df = clustered_df.withColumn("rank", row_number().over(windowSpec))
    representative_df = ranked_df.filter(col("rank") == 1).select(
        col("vx"), col("vy"), col("vz"), 
        col("id").alias("rep_id"), 
        col("x").alias("rep_x"), 
        col("y").alias("rep_y"), 
        col("z").alias("rep_z")
    )
    
    return clustered_df, representative_df


def create_vertex_mapping(clustered_df, representative_df):
    """원본 정점 ID를 대표 정점 ID로 매핑"""
    mapping_df = clustered_df.join(representative_df, ["vx", "vy", "vz"]) \
        .select(col("id").alias("old_id"), col("rep_id"))
    
    id_map = {row['old_id']: row['rep_id'] for row in mapping_df.collect()}
    
    rep_vertices_rows = representative_df.collect()
    rep_coords_map = {row['rep_id']: [row['rep_x'], row['rep_y'], row['rep_z']] 
                      for row in rep_vertices_rows}
    
    return id_map, rep_coords_map


def reconstruct_mesh(mesh, id_map, rep_coords_map):
    """메쉬 재구성"""
    simplified_faces = []
    
    for face in mesh.faces:
        v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
        
        if v0 in id_map and v1 in id_map and v2 in id_map:
            nv0 = id_map[v0]
            nv1 = id_map[v1]
            nv2 = id_map[v2]
            
            if nv0 != nv1 and nv1 != nv2 and nv2 != nv0:
                simplified_faces.append([nv0, nv1, nv2])
    
    # 최종 정점 및 면 생성
    final_vertices = []
    final_faces = []
    final_id_map = {}
    
    for i, (rep_id, coords) in enumerate(rep_coords_map.items()):
        final_id_map[rep_id] = i
        final_vertices.append(coords)
    
    for face in simplified_faces:
        final_faces.append([final_id_map[face[0]], final_id_map[face[1]], final_id_map[face[2]]])
    
    return final_vertices, final_faces


def save_simplified_mesh(final_vertices, final_faces, cluster_output, mesh):
    """단순화된 메쉬 저장"""
    if final_vertices and final_faces:
        cluster_mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces)
        cluster_mesh.export(cluster_output)
        print(f"Vertex Clustering 단순화 완료: {cluster_output}")
        print(f"  - 원본 정점 수: {len(mesh.vertices)}")
        print(f"  - 원본 면 갯수: {len(mesh.faces)}")
        print(f"  - 단순화된 정점 수: {len(final_vertices)} (비율: {len(final_vertices)/len(mesh.vertices)*100:.1f}%)")
        print(f"  - 단순화된 면 갯수: {len(final_faces)}")
    else:
        print("경고: Vertex Clustering 결과가 비어 있습니다.")


def main():
    """메인 함수"""
    # 환경 설정
    setup_environment()
    
    # Spark 세션 생성
    spark = create_spark_session()
    
    # 파일 경로 설정
    cat = 'bracket'
    input_path = fr"mesh_origin.obj"
    output_path = fr"mesh_test.obj"
    
    # 메쉬 로드
    mesh = load_mesh(input_path)
    
    # GraphFrame 데이터 생성
    vertices_df, edges_df = create_graph_dataframes(spark, mesh)
    
    # 삼각형 개수 계산
    triangle_counts = calculate_triangle_counts(vertices_df, edges_df)
    
    print("\n[추가 단순화] Vertex Clustering (Triangle Count 기반) 시작...")
    
    # 복셀 크기 계산 dim 변수로 복셀 사이즈 조절 가능
    voxel_size = calculate_voxel_size(mesh, dim=20)
    
    # 정점 클러스터링
    clustered_df, representative_df = cluster_vertices(triangle_counts, vertices_df, voxel_size)
    
    # 정점 매핑 생성
    id_map, rep_coords_map = create_vertex_mapping(clustered_df, representative_df)
    
    # 메쉬 재구성
    final_vertices, final_faces = reconstruct_mesh(mesh, id_map, rep_coords_map)
    
    # 결과 저장
    save_simplified_mesh(final_vertices, final_faces, output_path, mesh)


if __name__ == "__main__":
    main()
