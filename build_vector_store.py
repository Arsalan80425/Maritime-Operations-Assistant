"""
Build Vector Store Script - Ollama Version
Creates embeddings for RAG using Ollama (no API key needed)
Run this once before starting the application
"""

import os
import sys
from utils.data_loader import get_data_loader
from utils.vector_store import get_vector_store

def main():
    print("=" * 60)
    print("Maritime Operations Assistant - Vector Store Builder")
    print("Using Ollama for embeddings (100% local, no API key)")
    print("=" * 60)
    print()
    
    # Check if Ollama is running
    print("🔍 Checking Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running and accessible")
        else:
            print("❌ Ollama is not responding correctly")
            print("   Please start Ollama: ollama serve")
            sys.exit(1)
    except Exception as e:
        print("❌ Cannot connect to Ollama")
        print("   Make sure Ollama is running: ollama serve")
        print(f"   Error: {str(e)}")
        sys.exit(1)
    
    print()
    
    # Check for data files
    print("📁 Checking for data files...")
    data_dir = "data"
    required_files = ["Shipments.csv", "Port_Data_Clean.csv", "Daily_Report.csv"]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ Found: {file}")
        else:
            print(f"   ❌ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print()
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print(f"   Please place them in the '{data_dir}/' directory")
        sys.exit(1)
    
    print()
    
    # Load data
    print("📊 Loading maritime data...")
    try:
        data_loader = get_data_loader(data_dir)
        data = {
            'shipments': data_loader.shipments,
            'port_data': data_loader.port_data,
            'daily_report': data_loader.daily_report
        }
        print(f"   ✅ Loaded {len(data['shipments'])} shipments")
        print(f"   ✅ Loaded {len(data['port_data'])} ports")
        print(f"   ✅ Loaded {len(data['daily_report'])} daily reports")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        sys.exit(1)
    
    print()
    
    # Build vector store
    print("🔨 Building vector store with Ollama embeddings...")
    print("   This may take 5-10 minutes depending on your hardware...")
    print("   Using model: mistral")
    print()
    
    try:
        vector_store = get_vector_store(data_dir=data_dir, persist_dir="vector_store")
        vector_store.build_vector_store(data)
        
        print()
        print("=" * 60)
        print("✅ SUCCESS! Vector store created successfully")
        print("=" * 60)
        print()
        print("You can now run the application:")
        print("   streamlit run app.py")
        print()
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ERROR: Failed to build vector store")
        print("=" * 60)
        print(f"Error details: {str(e)}")
        print()
        print("Troubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull the model: ollama pull mistral")
        print("3. Check if you have enough disk space")
        print("4. Try deleting 'vector_store/' folder and running again")
        sys.exit(1)

if __name__ == "__main__":
    main()
