"""
Tracking Agent with RAG Support - FIXED VERSION
Handles container tracking with semantic search capabilities
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.data_loader import get_data_loader
from typing import Dict, Optional, List
from functools import lru_cache
import json

class TrackingAgent:
    """Agent responsible for shipment tracking with RAG support"""
    
    def __init__(self, model_name: str = "mistral"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            base_url="http://localhost:11434"
        )
        self.data_loader = get_data_loader()
        self.vector_store = None  # Will be set by orchestrator
        
        # Standard tracking prompt
        self.tracking_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a maritime tracking specialist. 
            Provide accurate, concise information about container shipments.
            Use the data provided to answer questions about:
            - Container locations
            - Shipment status
            - Estimated arrival times
            - Delay information
            
            Always be precise and include relevant details."""),
            ("human", "{query}\n\nData: {data}")
        ])
        
        # RAG-enhanced prompt
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a maritime tracking specialist with access to historical data.
            Provide comprehensive tracking information using:
            1. Current shipment data
            2. Historical context from knowledge base
            
            Include relevant patterns, similar cases, and contextual insights.
            Always cite specific data points."""),
            ("human", "{query}\n\nCurrent Data: {data}\n\nHistorical Context: {context}")
        ])
        
        self.chain = self.tracking_prompt | self.llm | StrOutputParser()
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        
        # Query cache for frequently accessed containers
        self._query_cache = {}
        self._cache_size = 100
    
    def _get_rag_context(self, query: str, k: int = 3) -> str:
        """Get relevant context from vector store with caching"""
        if not self.vector_store or not self.vector_store.vector_store:
            return "No historical context available"
        
        # Simple cache key based on query
        cache_key = f"rag_{hash(query)}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            
            if not results:
                return "No relevant historical data found"
            
            context_parts = []
            for i, doc in enumerate(results, 1):
                doc_type = doc.metadata.get('type', 'unknown')
                context_parts.append(f"[Reference {i} - {doc_type}]:\n{doc.page_content[:300]}...\n")
            
            context = "\n".join(context_parts)
            
            # Cache result
            if len(self._query_cache) >= self._cache_size:
                # Remove oldest entry
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[cache_key] = context
            
            return context
            
        except Exception as e:
            return f"Context retrieval error: {str(e)}"
    
    def _use_rag(self) -> bool:
        """Check if RAG is available"""
        return self.vector_store is not None and self.vector_store.vector_store is not None
    
    def track_container(self, container_id: str) -> str:
        """Track specific container by ID with RAG enhancement"""
        try:
            container = self.data_loader.get_container_info(container_id)
            
            if not container:
                return f"❌ Container {container_id} not found in the system."
            
            query = f"Provide tracking details for container {container_id}"
            data = json.dumps(container, indent=2, default=str)
            
            # Use RAG if available
            if self._use_rag():
                rag_query = f"container {container_id} {container['Port_Name']} {container['Status']} {container['Cargo_Type']}"
                context = self._get_rag_context(rag_query, k=3)
                
                response = self.rag_chain.invoke({
                    "query": query,
                    "data": data,
                    "context": context
                })
                return f"📦 **RAG-Enhanced Tracking:**\n\n{response}"
            else:
                response = self.chain.invoke({
                    "query": query,
                    "data": data
                })
                return response
            
        except Exception as e:
            return f"Error tracking container: {str(e)}"
    
    def get_location(self, container_id: str) -> str:
        """Get current location of container"""
        try:
            container = self.data_loader.get_container_info(container_id)
            
            if not container:
                return f"Container {container_id} not found."
            
            status = container['Status']
            port = container['Port_Name']
            
            if status == 'Arrived':
                return f"📍 Container {container_id} has ARRIVED at {port}"
            elif status == 'In Transit':
                return f"🚢 Container {container_id} is IN TRANSIT to {port}"
            elif status == 'Delayed':
                delay = container['Delay_Hours']
                return f"⏰ Container {container_id} is DELAYED at {port} by {delay} hours"
            else:
                return f"Container {container_id} status: {status} at {port}"
                
        except Exception as e:
            return f"Error getting location: {str(e)}"
    
    def get_eta(self, container_id: str) -> str:
        """Get estimated time of arrival with RAG context"""
        try:
            container = self.data_loader.get_container_info(container_id)
            
            if not container:
                return f"Container {container_id} not found."
            
            eta = container['ETA']
            port = container['Port_Name']
            status = container['Status']
            delay = container['Delay_Hours']
            
            eta_data = {
                'container_id': container_id,
                'port': port,
                'eta': str(eta),
                'status': status,
                'delay_hours': delay
            }
            
            query = f"Provide ETA information for container {container_id}"
            data = json.dumps(eta_data, indent=2)
            
            # Add historical ETA accuracy context if RAG available
            if self._use_rag():
                context = self._get_rag_context(f"ETA accuracy {port} {container['Cargo_Type']}", k=2)
                response = self.rag_chain.invoke({
                    "query": query,
                    "data": data,
                    "context": context
                })
                return response
            else:
                response = self.chain.invoke({
                    "query": query,
                    "data": data
                })
                return response
            
        except Exception as e:
            return f"Error getting ETA: {str(e)}"
    
    def find_containers_by_port(self, port_name: str) -> str:
        """Find all containers at specific port using RAG"""
        try:
            if not self._use_rag():
                # Fallback to data loader
                shipments = self.data_loader.shipments
                port_containers = shipments[
                    shipments['Port_Name'].str.upper() == port_name.upper()
                ]
                
                if port_containers.empty:
                    return f"No containers found at port {port_name}"
                
                containers = port_containers.head(10).to_dict('records')
                query = f"List containers currently at {port_name}"
                data = json.dumps(containers, indent=2, default=str)
                
                return self.chain.invoke({"query": query, "data": data})
            
            # Use RAG for semantic search
            results = self.vector_store.similarity_search(
                f"containers at {port_name} port shipments",
                k=10
            )
            
            shipment_docs = [
                doc for doc in results 
                if doc.metadata.get('type') == 'shipment' and 
                port_name.upper() in doc.metadata.get('port', '').upper()
            ]
            
            if not shipment_docs:
                return f"No containers found at port {port_name}"
            
            containers = []
            for doc in shipment_docs[:10]:
                containers.append({
                    'container_id': doc.metadata.get('container_id'),
                    'status': doc.metadata.get('status'),
                    'port': doc.metadata.get('port'),
                    'cargo_type': doc.metadata.get('cargo_type')
                })
            
            query = f"Summarize containers at {port_name}"
            data = json.dumps(containers, indent=2)
            
            response = self.chain.invoke({
                "query": query,
                "data": data
            })
            
            return f"📦 **Found {len(containers)} containers at {port_name}**\n\n{response}"
            
        except Exception as e:
            return f"Error finding containers: {str(e)}"
    
    def get_delayed_containers(self, port_name: Optional[str] = None) -> str:
        """Get all delayed containers with RAG insights"""
        try:
            delayed = self.data_loader.get_delayed_shipments(port_name)
            
            if delayed.empty:
                location = f"at {port_name}" if port_name else "in the system"
                return f"No delayed containers found {location}"
            
            delayed_list = delayed.head(15).to_dict('records')
            
            query = f"Summarize delayed containers" + (f" at {port_name}" if port_name else "")
            data = json.dumps(delayed_list, indent=2, default=str)
            
            # Add historical delay patterns if RAG available
            if self._use_rag() and port_name:
                context = self._get_rag_context(f"delay patterns {port_name} causes trends", k=4)
                response = self.rag_chain.invoke({
                    "query": query,
                    "data": data,
                    "context": context
                })
                return f"⏰ **Delayed Containers Analysis:**\n\n{response}"
            else:
                response = self.chain.invoke({
                    "query": query,
                    "data": data
                })
                return response
            
        except Exception as e:
            return f"Error getting delayed containers: {str(e)}"
    
    def search_with_rag(self, query: str) -> str:
        """
        NEW: RAG-powered semantic search for complex tracking queries
        This is the method the orchestrator expects
        """
        if not self._use_rag():
            return "RAG search not available. Please provide a specific container ID or port name."
        
        try:
            # Perform semantic search
            results = self.vector_store.similarity_search(query, k=8)
            
            if not results:
                return "No relevant tracking information found for your query."
            
            # Filter for tracking-relevant data
            tracking_docs = [
                doc for doc in results 
                if doc.metadata.get('type') in ['shipment', 'daily_report', 'port']
            ]
            
            if not tracking_docs:
                return "No relevant shipment data found."
            
            # Prepare context
            context_parts = []
            for i, doc in enumerate(tracking_docs[:6], 1):
                doc_type = doc.metadata.get('type', 'unknown')
                context_parts.append(f"[Result {i} - {doc_type}]\n{doc.page_content[:250]}...")
            
            context = "\n\n".join(context_parts)
            
            # Generate response
            search_query = f"Provide tracking information and insights for: {query}"
            
            response = self.rag_chain.invoke({
                "query": search_query,
                "data": f"Found {len(tracking_docs)} relevant records",
                "context": context
            })
            
            return f"🔍 **Semantic Search Results:**\n\n{response}\n\n📚 *Based on {len(tracking_docs)} relevant records*"
            
        except Exception as e:
            return f"Error performing semantic search: {str(e)}"
    
    def clear_cache(self):
        """Clear query cache"""
        self._query_cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            'cached_queries': len(self._query_cache),
            'cache_size_limit': self._cache_size,
            'rag_available': self._use_rag()
        }

# Singleton instance
_tracking_agent = None

def get_tracking_agent() -> TrackingAgent:
    """Get or create tracking agent instance"""
    global _tracking_agent
    if _tracking_agent is None:
        _tracking_agent = TrackingAgent()
    return _tracking_agent