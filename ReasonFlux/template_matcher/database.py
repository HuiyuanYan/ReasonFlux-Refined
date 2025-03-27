import chromadb
from chromadb.api import ClientAPI
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, List

from ReasonFlux.template_matcher.service import (
    EmbeddingService,
    OpenAIEmbeddingService,
    OllamaEmbeddingService,
    JinaAIEmbeddingService
)
from ReasonFlux.utils.common import get_uuid, logger


class HierarchicalVectorDatabase(BaseModel):
    """
    A hierarchical vector database for storing and querying hierarchical data.

    This class provides functionality for building a hierarchical structure of vectors,
    querying the database at different levels, and managing collections. It uses an
    embedding service to convert text into vectors and ChromaDB as the underlying
    vector database.

    Attributes:
        data_dir (str): The directory where the vector database is stored.
        collections (dict): The collections of the vector database.
        max_level (int): The maximum number of levels in the hierarchical vector database.
        embedding_service (EmbeddingService): The embedding service used by the vector database.
        chroma_client (ClientAPI): The ChromaDB client used by the vector database.
        embedding_params (dict): The parameters of the embedding function.
        persist (bool): Whether to persist the database.
    """
    data_dir:str = Field(
        default="data",
        description="The directory where the vector database is stored"
    )
    
    collections:dict = Field(
        default_factory=dict,
        description="The collections of the vector database"
    )

    max_level: int = Field(
        default=0,
        description="The maximum number of levels in the hierarchical vector database"
    )

    embedding_service: EmbeddingService = Field(
        default=None,
        description="The embedding service used by the vector database"
    )

    chroma_client: ClientAPI = Field(
        default=None,
        description="The ChromaDB client used by the vector database"
    )

    embedding_params: dict = Field(
        default={
            "api_key": "sh-xx",
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "text-embedding-v3",
            "provider": "openai"
        },
        description="The parameters of the embedding function"
    )

    persist: bool = Field(
        default=True,
        description="Whether to persist the database"
    )

    class Config:
        arbitrary_types_allowed: bool = True

    @model_validator(mode="after")
    def initialize_database(self) -> "HierarchicalVectorDatabase":
        if self.embedding_service is None or not isinstance(self.embedding_service, EmbeddingService):
            match self.embedding_params["provider"]:
                case "openai":
                    self.embedding_service = OpenAIEmbeddingService(
                        api_key=self.embedding_params["api_key"],
                        api_base=self.embedding_params["api_base"],
                        model_name=self.embedding_params["model"]
                    )
                case "jina":
                    self.embedding_service = JinaAIEmbeddingService(
                        api_key=self.embedding_params["api_key"],
                        model_name=self.embedding_params["model"]
                    )
                case "ollama":
                    self.embedding_service = OllamaEmbeddingService(
                        url=self.embedding_params["api_base"],
                        model_name=self.embedding_params["model"]
                    )
                case _:
                    raise ValueError("Invalid embedding provider")
        if self.chroma_client is None or not isinstance(self.chroma_client, ClientAPI):
            if self.persist:
                self.chroma_client = chromadb.PersistentClient(path = self.data_dir)
            else:
                self.chroma_client = chromadb.EphemeralClient()
            self._load_from_chroma_client()
        return self

    def _load_from_chroma_client(self):
        """
        Load the collections and set max_level from the ChromaDB client.
        """
        i = 0
        all_collections = self.chroma_client.list_collections()
        while True:
            collection_name = f"level_{i}"
            if collection_name in all_collections:
                self.collections[collection_name] = self.chroma_client.get_collection(collection_name)
            else:
                break
            i += 1
        self.max_level = i

    def add_recursive_dict(self, data: Dict[str, Any]):
        """
        Build the hierarchical database from a recursive dictionary.

        Args:
            data (Dict[str, Any]): The recursive dictionary to add to the database.
        """
        level = self._determine_depth(data)
        for i in range(level):
            collection_name = f"level_{i}"
            if collection_name not in self.collections:
                self._create_collection(collection_name)
        self.max_level = max(self.max_level, level)
        self._recursive_add(data, 0)

    def _determine_depth(self, data: Dict[str, Any], current_depth: int = 0) -> int:
        """
        Determine the depth of the hierarchical data.

        Args:
            data (Dict[str, Any]): The recursive dictionary.
            current_depth (int): Current depth in the recursion.

        Returns:
            int: The maximum depth of the data.
        """
        if not isinstance(data, dict):
            return current_depth
        max_depth = current_depth
        for key in data:
            max_depth = max(max_depth, self._determine_depth(data[key], current_depth + 1))
        return max_depth


    def _create_collection(self, collection_name: str):
        """
        Create a new collection in the database.
        :param collection_name: The name of the collection.
        """
        logger.info(f"Creating collection: {collection_name}")
        if collection_name not in self.collections:
            self.collections[collection_name] = self.chroma_client.get_or_create_collection(collection_name)
        logger.info(f"Collection created: {collection_name}")

    def _delete_collection(self, collection_name: str):
        """
        Delete a collection from the database.

        Args:
            collection_name (str): The name of the collection.
        """
        logger.info(f"Deleting collection: {collection_name}")
        if collection_name in self.collections:
            self.chroma_client.delete_collection(collection_name)
            del self.collections[collection_name]
        logger.info(f"Collection deleted: {collection_name}")

    def _recursive_add(self, data: Dict[str, Any], current_level: int, parent_id:str = None):
        """
        Recursively build the database.

        Args:
            data (Dict[str, Any]): The recursive dictionary.
            current_level (int): Current level in the hierarchy.
            parent_id (str): The parent ID for the current level.
        """
        documents = []
        node_ids = []
        embeddings = []
        meta_data_list = [] 

        for key, value in data.items():
            current_meta_data = []
            current_node_id = str(get_uuid())
            current_embedding = self.embedding_service.encode(key)

            if isinstance(value, dict):
                self._recursive_add(value, current_level + 1, current_node_id)
                current_meta_data ={
                    "parent": parent_id or "",
                    "depth": current_level,
                    "data": ""
                }
            else:
                current_meta_data ={
                    "parent": parent_id or "",
                    "depth": current_level,
                    "data": str(value)
                }
            
            documents.append(key)
            node_ids.append(current_node_id)
            embeddings.append(current_embedding)
            meta_data_list.append(current_meta_data)
        
        # 向对应层添加数据
        self.collections[f"level_{current_level}"].add(
            documents=documents,
            ids=node_ids,
            embeddings=embeddings,
            metadatas=meta_data_list
        )
        return
    
    def hierarchical_search(
        self,
        queries: list[str],
        top_k_per_level: list[int],
        weight_per_level: list[float],
        search_level: int = None,
        final_count: int = 1
    )-> List[Dict[str,Any]] | None:
        """
        Perform a hierarchical search across multiple levels of the database.

        This function traverses the hierarchical structure of the database, querying each level
        based on the provided queries and parameters. It combines results from each level using
        weighted distances and filters them based on parent-child relationships.

        Args:
            queries (list[str]): List of queries for each level.
            top_k_per_level (list[int]): Number of top results to retrieve from each level.
            weight_per_level (list[float]): Weights assigned to results from each level.
            search_level (int, optional): The maximum level to search. Defaults to self.max_level.
            final_count (int, optional): Number of final results to return. Defaults to 1.

        Returns:
            List[Dict[str, Any]] | None: List of top results with their metadata and distances, or None if an error occurs.
        """
        def _distance_to_similarity(distance: float) -> float:
            return 1 / (1 + distance)

        if search_level is None:
            logger.info(f"search level is None, using max level: {self.max_level}")
            search_level = self.max_level
        
        if search_level > self.max_level:
            logger.error(f"search level is out of range, max level is {self.max_level}")
            return None

        if len(queries) != len(top_k_per_level) or len(queries) != len(weight_per_level):
            logger.error("queries, top_k_per_level, weight_per_level should have the same length")
            return None
        
        parent_id_sim_list = []
        tmp_cand = []
        for search_idx in range(search_level):
            if tmp_cand:
                parent_id_sim_list = [
                    (cand["id"], cand["similarity"]) for cand in tmp_cand
                ]
                tmp_cand.clear()
            collection_name = f"level_{search_idx}"
            current_k = top_k_per_level[search_idx]
            current_weight = weight_per_level[search_idx]
            current_embedding = self.embedding_service.encode(queries[search_idx])
            
            if parent_id_sim_list:
                seen_ids = set()
                for parent_id, parent_sim in parent_id_sim_list:
                    query_res = self.collections[collection_name].query(
                        query_embeddings=[current_embedding],
                        n_results = current_k,
                        where={"parent": {"$eq": parent_id}}
                    )
                    for res_idx in range(0, len(query_res['ids'][0])):
                        if query_res['ids'][0][res_idx] in seen_ids:
                            continue
                        tmp_cand.append(
                            {
                                "doc": query_res['documents'][0][res_idx],
                                "id": query_res['ids'][0][res_idx],
                                "similarity": _distance_to_similarity(query_res['distances'][0][res_idx])*current_weight + parent_sim,
                                "meta_data": query_res['metadatas'][0][res_idx],
                            }
                        )
                        seen_ids.add(query_res['ids'][0][res_idx])
            else:
                seen_ids = set()
                query_res = self.collections[collection_name].query(
                    query_embeddings=[current_embedding],
                    n_results = current_k
                )
                for res_idx in range(0, len(query_res['ids'][0])):
                    if query_res['ids'][0][res_idx] in seen_ids:
                        continue
                    tmp_cand.append(
                        {
                            "doc": query_res['documents'][0][res_idx],
                            "id": query_res['ids'][0][res_idx],
                            "similarity":  _distance_to_similarity(query_res['distances'][0][res_idx])*current_weight,
                            "meta_data": query_res['metadatas'][0][res_idx],
                        }
                    )
                    seen_ids.add(query_res['ids'][0][res_idx])                
            pass

        final_res = sorted(tmp_cand, key=lambda x: x["similarity"], reverse=True)[:final_count]
        return final_res
    
    def clear(self):
        """
        Clear all collections in the database and reset the database state.
        """
        logger.info("Clearing the database...")
        for collection_name in list(self.collections.keys()):
            self._delete_collection(collection_name)
        self.max_level = 0
        logger.info("Database cleared successfully.")