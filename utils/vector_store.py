"""
Enhanced Vector Store with Query Caching and Optimization
Reduces redundant embedding calls and improves performance
"""

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict, Tuple
import pandas as pd
import os
from pathlib import Path
import pickle
import json
import hashlib
from functools import lru_cache
from datetime import datetime

class MaritimeVectorStore:
    """Vector store with comprehensive caching and optimization"""
    
    def __init__(self, data_dir: str = "data", persist_dir: str = "vector_store", model: str = "mistral"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.model = model
        
        # Use Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=model,
            base_url="http://localhost:11434"
        )
        self.vector_store = None
        self.metadata_cache = {}
        
        # Query cache: stores (query_hash -> results)
        self._query_cache = {}
        self._query_cache_max_size = 500  # Increased from 200
        
        # Popular queries tracking for intelligent eviction
        self._popular_queries = {}
        
        # Cache TTL
        self._cache_ttl_seconds = 3600  # 1 hour
        
        # Search statistics
        self._search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_reset': datetime.now().isoformat()
        }
    
    def _hash_query(self, query: str, k: int) -> str:
        """Create hash for query caching"""
        # Normalize query for better cache hits
        normalized = query.lower().strip()
        cache_key = f"{normalized}_{k}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in cache_entry:
            return False
        
        age = datetime.now() - cache_entry['timestamp']
        return age.total_seconds() < self._cache_ttl_seconds
    
    def create_documents_from_data(self, data: Dict[str, pd.DataFrame]) -> List[Document]:
        """Convert dataframes to LangChain documents with rich metadata"""
        documents = []
        stats = {"shipments": 0, "ports": 0, "daily_reports": 0}
        
        print("\n📄 Creating documents from data...")
        
        # Process Shipments
        print("   Processing shipments...")
        for idx, row in data['shipments'].iterrows():
            content = f"""Container ID: {row['Container_ID']}
Port: {row['Port_Name']}
Status: {row['Status']}
ETA: {row['ETA']}
Delay: {row['Delay_Hours']} hours
Cargo Type: {row['Cargo_Type']}

This is a {row['Cargo_Type']} shipment currently {row['Status']} at {row['Port_Name']} port.
{'The shipment is delayed by ' + str(row['Delay_Hours']) + ' hours.' if row['Delay_Hours'] > 0 else 'The shipment is on schedule.'}
Expected arrival time is {row['ETA']}.
"""
            
            documents.append(Document(
                page_content=content,
                metadata={
                    'type': 'shipment',
                    'container_id': row['Container_ID'],
                    'port': row['Port_Name'],
                    'status': row['Status'],
                    'cargo_type': row['Cargo_Type'],
                    'delay_hours': float(row['Delay_Hours'])
                }
            ))
            stats["shipments"] += 1
        
        print(f"      ✅ Created {stats['shipments']} shipment documents")
        
        # Process Port Data
        print("   Processing port data...")
        for idx, row in data['port_data'].iterrows():
            content = f"""Port Name: {row['Port Name']}
Country: {row['Country']}
UN Code: {row['UN Code']}
Vessels in Port: {row['Vessels in Port']}
Traffic Category: {row['Traffic Category']}
Port Activity Index: {row['Port Activity Index']}
Active Ratio: {row['Active Ratio']}
Traffic Density: {row['Traffic Density']}

{row['Port Name']} is a {row['Traffic Category']} traffic port located in {row['Country']}.
Currently, there are {row['Vessels in Port']} vessels in this port.
The port has an activity index of {row['Port Activity Index']} and traffic density of {row['Traffic Density']}.
"""
            
            documents.append(Document(
                page_content=content,
                metadata={
                    'type': 'port',
                    'port_name': row['Port Name'],
                    'country': row['Country'],
                    'traffic_category': row['Traffic Category'],
                    'vessels_in_port': int(row['Vessels in Port']),
                    'un_code': row['UN Code']
                }
            ))
            stats["ports"] += 1
        
        print(f"      ✅ Created {stats['ports']} port documents")
        
        # Process Daily Reports
        print("   Processing daily reports...")
        for idx, row in data['daily_report'].iterrows():
            content = f"""Date: {row['Date']}
Port: {row['Port_Name']}
Vessels in Port: {row['Vessels_in_Port']}
Average Delay: {row['Avg_Delay']} hours
Weather: {row['Weather']}
Operational Status: {row['Remarks']}

On {row['Date']}, {row['Port_Name']} port had {row['Vessels_in_Port']} vessels.
Weather conditions were {row['Weather']}.
Average delay was {row['Avg_Delay']} hours.
Port status: {row['Remarks']}.
"""
            
            documents.append(Document(
                page_content=content,
                metadata={
                    'type': 'daily_report',
                    'port': row['Port_Name'],
                    'date': str(row['Date']),
                    'weather': row['Weather'],
                    'avg_delay': float(row['Avg_Delay']),
                    'vessels': int(row['Vessels_in_Port']),
                    'remarks': row['Remarks']
                }
            ))
            stats["daily_reports"] += 1
        
        print(f"      ✅ Created {stats['daily_reports']} daily report documents")
        
        total_docs = sum(stats.values())
        print(f"\n   📊 Total documents created: {total_docs}")
        
        # Cache statistics
        self.metadata_cache = {
            'total_documents': total_docs,
            'document_types': stats,
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'model': self.model
        }
        
        return documents
    
    def build_vector_store(self, data: Dict[str, pd.DataFrame]):
        """Build and persist vector store with verification"""
        print("\n" + "="*60)
        print("🗃️ Building Vector Store with Ollama Embeddings")
        print("="*60)
        
        # Create documents
        documents = self.create_documents_from_data(data)
        
        if not documents:
            raise ValueError("No documents created from data!")
        
        print(f"\n📄 Creating FAISS index with {len(documents)} documents...")
        print("   This may take 5-10 minutes depending on your hardware...")
        
        try:
            # Create FAISS vector store with Ollama embeddings
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            print("   ✅ FAISS index created successfully")
            
        except Exception as e:
            print(f"   ❌ Error creating FAISS index: {str(e)}")
            raise
        
        # Save to disk
        print(f"\n💾 Saving vector store to {self.persist_dir}...")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        try:
            self.vector_store.save_local(self.persist_dir)
            print("   ✅ Vector store saved successfully")
        except Exception as e:
            print(f"   ❌ Error saving vector store: {str(e)}")
            raise
        
        # Save metadata with caching info
        metadata_path = Path(self.persist_dir) / "metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata_cache, f, indent=2)
            print("   ✅ Metadata saved")
        except Exception as e:
            print(f"   ⚠️ Warning: Could not save metadata: {str(e)}")
        
        # Save initial cache stats
        self._save_cache_stats()
        
        # Verify files were created
        self._verify_saved_files()
        
        # Test the vector store
        self._test_vector_store()
        
        print("\n" + "="*60)
        print("✅ Vector Store Build Complete!")
        print("="*60)
    
    def _verify_saved_files(self):
        """Verify that all required files were created"""
        print(f"\n🔍 Verifying saved files in {self.persist_dir}...")
        
        required_files = ['index.faiss', 'index.pkl']
        all_exist = True
        
        for filename in required_files:
            filepath = Path(self.persist_dir) / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"   ✅ {filename}: {size:,} bytes")
                
                # Warn if files are suspiciously small
                if filename == 'index.faiss' and size < 1000:
                    print(f"      ⚠️ File seems small - may be incomplete")
                    all_exist = False
                elif filename == 'index.pkl' and size < 100:
                    print(f"      ⚠️ File seems small - may be incomplete")
                    all_exist = False
            else:
                print(f"   ❌ {filename}: NOT FOUND")
                all_exist = False
        
        # Check metadata
        metadata_path = Path(self.persist_dir) / "metadata.json"
        if metadata_path.exists():
            print(f"   ✅ metadata.json: {metadata_path.stat().st_size} bytes")
        
        if not all_exist:
            raise ValueError("Vector store files are missing or incomplete!")
    
    def _test_vector_store(self):
        """Test the vector store with sample queries"""
        print("\n🧪 Testing vector store...")
        
        test_queries = [
            ("container CNT10000", "shipment"),
            ("Shanghai port", "port"),
            ("delays", "daily_report")
        ]
        
        all_tests_passed = True
        
        for query, expected_type in test_queries:
            try:
                results = self.vector_store.similarity_search(query, k=2)
                
                if results:
                    found_type = results[0].metadata.get('type', 'unknown')
                    if expected_type in found_type or found_type in expected_type:
                        print(f"   ✅ Query '{query}': Found {len(results)} results (type: {found_type})")
                    else:
                        print(f"   ⚠️ Query '{query}': Found {found_type}, expected {expected_type}")
                else:
                    print(f"   ❌ Query '{query}': No results found")
                    all_tests_passed = False
                    
            except Exception as e:
                print(f"   ❌ Query '{query}': Error - {str(e)}")
                all_tests_passed = False
        
        if not all_tests_passed:
            print("   ⚠️ Some tests failed - vector store may not work properly")
        else:
            print("   ✅ All tests passed!")
    
    def preload_common_queries(self):
        """Preload cache with common queries for faster responses"""
        common_queries = [
            "container delays",
            "port congestion",
            "shipment status",
            "weather impact",
            "cargo performance",
            "ETA information",
            "tracking updates",
            "delay patterns",
            "port operations",
            "vessel traffic"
        ]
        
        print("🔥 Preloading common queries into cache...")
        for query in common_queries:
            try:
                self.similarity_search(query, k=3, use_cache=True)
            except:
                pass  # Silently skip errors during preload
        
        print(f"✅ Preloaded {len(common_queries)} common queries")
    
    def load_vector_store(self):
        """Load existing vector store with verification"""
        vector_store_path = Path(self.persist_dir)
        
        if not vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store directory not found: {self.persist_dir}\n"
                "Please run: python build_vector_store.py"
            )
        
        # Check required files
        index_faiss = vector_store_path / "index.faiss"
        index_pkl = vector_store_path / "index.pkl"
        
        if not index_faiss.exists() or not index_pkl.exists():
            missing = []
            if not index_faiss.exists():
                missing.append("index.faiss")
            if not index_pkl.exists():
                missing.append("index.pkl")
            
            raise FileNotFoundError(
                f"Required vector store files missing: {', '.join(missing)}\n"
                "Please run: python build_vector_store.py"
            )
        
        print(f"📂 Loading vector store from {self.persist_dir}...")
        print(f"   index.faiss: {index_faiss.stat().st_size:,} bytes")
        print(f"   index.pkl: {index_pkl.stat().st_size:,} bytes")
        
        try:
            self.vector_store = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata if available
            metadata_path = vector_store_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata_cache = json.load(f)
                    print(f"   📊 Loaded metadata: {self.metadata_cache.get('total_documents', 'unknown')} documents")
            
            # Load cache stats if available
            self._load_cache_stats()
            
            # Verify with a test search
            test_results = self.vector_store.similarity_search("test", k=1)
            if test_results:
                print(f"   ✅ Vector store loaded and verified")
                print(f"   🔍 Cache: {len(self._query_cache)} queries cached")
            else:
                print(f"   ⚠️ Vector store loaded but seems empty")
            
            # Preload common queries for faster responses
            self.preload_common_queries()
                
        except Exception as e:
            raise Exception(
                f"Error loading vector store: {str(e)}\n"
                "The vector store may be corrupted. Try rebuilding with: python build_vector_store.py"
            )
    
    def similarity_search(self, query: str, k: int = 5, use_cache: bool = True) -> List[Document]:
        """
        Perform similarity search with intelligent caching
        
        Args:
            query: Search query
            k: Number of results
            use_cache: Whether to use query cache (default: True)
        """
        if self.vector_store is None:
            print("⚠️ Vector store not loaded, attempting to load...")
            self.load_vector_store()
        
        if self.vector_store is None:
            return []
        
        self._search_stats['total_searches'] += 1
        
        # Check cache if enabled
        if use_cache:
            query_hash = self._hash_query(query, k)
            
            if query_hash in self._query_cache:
                cache_entry = self._query_cache[query_hash]
                
                # Check if cache is still valid
                if self._is_cache_valid(cache_entry):
                    self._search_stats['cache_hits'] += 1
                    
                    # Track popular queries for intelligent eviction
                    self._popular_queries[query_hash] = self._popular_queries.get(query_hash, 0) + 1
                    
                    return cache_entry['results']
        
        # Cache miss - perform actual search
        self._search_stats['cache_misses'] += 1
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            
            # Cache the results
            if use_cache and results:
                query_hash = self._hash_query(query, k)
                
                # Intelligent LRU eviction - remove least popular if full
                if len(self._query_cache) >= self._query_cache_max_size:
                    if self._popular_queries:
                        # Remove least popular entry
                        least_popular = min(self._popular_queries, key=self._popular_queries.get)
                        if least_popular in self._query_cache:
                            del self._query_cache[least_popular]
                            del self._popular_queries[least_popular]
                    else:
                        # Fallback: remove oldest
                        oldest_key = next(iter(self._query_cache))
                        del self._query_cache[oldest_key]
                
                self._query_cache[query_hash] = {
                    'results': results,
                    'timestamp': datetime.now()
                }
                self._popular_queries[query_hash] = 1
            
            return results
            
        except Exception as e:
            print(f"❌ Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[Tuple]:
        """Perform similarity search with relevance scores (with caching)"""
        if self.vector_store is None:
            self.load_vector_store()
        
        if self.vector_store is None:
            return []
        
        # Check cache for scored searches too
        query_hash = self._hash_query(f"scored_{query}", k)
        
        if query_hash in self._query_cache:
            cache_entry = self._query_cache[query_hash]
            if self._is_cache_valid(cache_entry):
                self._search_stats['cache_hits'] += 1
                return cache_entry['results']
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Cache results
            if results and len(self._query_cache) < self._query_cache_max_size:
                self._query_cache[query_hash] = {
                    'results': results,
                    'timestamp': datetime.now()
                }
            
            return results
        except Exception as e:
            print(f"❌ Error during scored search: {str(e)}")
            return []
    
    def clear_cache(self):
        """Clear query cache"""
        self._query_cache.clear()
        self._popular_queries.clear()
        self._search_stats['cache_hits'] = 0
        self._search_stats['cache_misses'] = 0
        self._search_stats['last_reset'] = datetime.now().isoformat()
        print("✅ Query cache cleared")
    
    def _save_cache_stats(self):
        """Save cache statistics to disk"""
        stats_path = Path(self.persist_dir) / "cache_stats.json"
        try:
            with open(stats_path, 'w') as f:
                json.dump(self._search_stats, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save cache stats: {str(e)}")
    
    def _load_cache_stats(self):
        """Load cache statistics from disk"""
        stats_path = Path(self.persist_dir) / "cache_stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    self._search_stats = json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load cache stats: {str(e)}")
    
    def get_statistics(self) -> dict:
        """Get comprehensive vector store and cache statistics"""
        hit_rate = 0
        if self._search_stats['total_searches'] > 0:
            hit_rate = (self._search_stats['cache_hits'] / self._search_stats['total_searches']) * 100
        
        stats = {
            'loaded': self.vector_store is not None,
            'persist_dir': str(self.persist_dir),
            'model': self.model,
            'cache': {
                'cached_queries': len(self._query_cache),
                'cache_max_size': self._query_cache_max_size,
                'total_searches': self._search_stats['total_searches'],
                'cache_hits': self._search_stats['cache_hits'],
                'cache_misses': self._search_stats['cache_misses'],
                'hit_rate': f"{hit_rate:.1f}%",
                'popular_queries': len(self._popular_queries)
            }
        }
        
        if self.metadata_cache:
            stats.update(self.metadata_cache)
        
        # Check file sizes
        vector_store_path = Path(self.persist_dir)
        if vector_store_path.exists():
            index_faiss = vector_store_path / "index.faiss"
            index_pkl = vector_store_path / "index.pkl"
            
            if index_faiss.exists():
                stats['faiss_size_bytes'] = index_faiss.stat().st_size
            if index_pkl.exists():
                stats['pkl_size_bytes'] = index_pkl.stat().st_size
        
        return stats
    
    def optimize_cache(self):
        """Optimize cache by removing expired and rarely used entries"""
        current_time = datetime.now()
        expired_keys = []
        
        # Remove expired entries
        for key, entry in self._query_cache.items():
            if 'timestamp' in entry:
                age = current_time - entry['timestamp']
                if age.total_seconds() > self._cache_ttl_seconds:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self._query_cache[key]
            if key in self._popular_queries:
                del self._popular_queries[key]
        
        if expired_keys:
            print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
        
        # If hit rate is low, refresh cache with common queries
        if self._search_stats['total_searches'] > 50:
            hit_rate = self._search_stats['cache_hits'] / self._search_stats['total_searches']
            if hit_rate < 0.3:  # Less than 30% hit rate
                print(f"⚠️ Low cache hit rate ({hit_rate*100:.1f}%), refreshing cache...")
                self.preload_common_queries()

# Singleton instance
_vector_store = None

def get_vector_store(data_dir: str = "data", persist_dir: str = "vector_store") -> MaritimeVectorStore:
    """Get or create vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = MaritimeVectorStore(data_dir, persist_dir)
    return _vector_store