"""
Data Loader Utility - Ollama Version
Loads and preprocesses maritime datasets
No OpenAI dependencies!
"""

import pandas as pd
import os
from typing import Dict, Optional
from datetime import datetime

class DataLoader:
    """Handles loading and preprocessing of maritime data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.daily_report = None
        self.port_data = None
        self.shipments = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets"""
        print("Loading maritime datasets...")
        
        # Load Daily Report
        daily_path = os.path.join(self.data_dir, "Daily_Report.csv")
        self.daily_report = pd.read_csv(daily_path)
        self.daily_report['Date'] = pd.to_datetime(self.daily_report['Date'])
        
        # Load Port Data
        port_path = os.path.join(self.data_dir, "Port_Data_Clean.csv")
        self.port_data = pd.read_csv(port_path)
        
        # Load Shipments
        shipments_path = os.path.join(self.data_dir, "Shipments.csv")
        self.shipments = pd.read_csv(shipments_path)
        self.shipments['ETA'] = pd.to_datetime(self.shipments['ETA'])
        
        print(f"✓ Loaded {len(self.daily_report)} daily reports")
        print(f"✓ Loaded {len(self.port_data)} port records")
        print(f"✓ Loaded {len(self.shipments)} shipments")
        
        return {
            'daily_report': self.daily_report,
            'port_data': self.port_data,
            'shipments': self.shipments
        }
    
    def get_container_info(self, container_id: str) -> Optional[Dict]:
        """Get specific container information"""
        if self.shipments is None:
            self.load_all_data()
            
        container = self.shipments[self.shipments['Container_ID'] == container_id]
        
        if container.empty:
            return None
            
        return container.iloc[0].to_dict()
    
    def get_port_delays(self, port_name: str, days: int = 7) -> pd.DataFrame:
        """Get port delays for specified days"""
        if self.daily_report is None:
            self.load_all_data()
            
        port_data = self.daily_report[
            self.daily_report['Port_Name'].str.upper() == port_name.upper()
        ]
        
        # Get last N days
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        recent_data = port_data[port_data['Date'] >= cutoff_date]
        
        return recent_data.sort_values('Date', ascending=False)
    
    def get_delayed_shipments(self, port_name: Optional[str] = None) -> pd.DataFrame:
        """Get all delayed shipments, optionally filtered by port"""
        if self.shipments is None:
            self.load_all_data()
            
        delayed = self.shipments[self.shipments['Status'] == 'Delayed']
        
        if port_name:
            delayed = delayed[
                delayed['Port_Name'].str.upper() == port_name.upper()
            ]
            
        return delayed
    
    def get_port_statistics(self, port_name: str) -> Optional[Dict]:
        """Get comprehensive port statistics"""
        if self.port_data is None:
            self.load_all_data()
            
        port = self.port_data[
            self.port_data['Port Name'].str.upper() == port_name.upper()
        ]
        
        if port.empty:
            return None
            
        return port.iloc[0].to_dict()
    
    def get_cargo_type_analysis(self) -> pd.DataFrame:
        """Analyze delays by cargo type"""
        if self.shipments is None:
            self.load_all_data()
            
        analysis = self.shipments.groupby('Cargo_Type').agg({
            'Delay_Hours': ['mean', 'sum', 'count'],
            'Container_ID': 'count'
        }).round(2)
        
        analysis.columns = ['Avg_Delay', 'Total_Delay', 'Delayed_Count', 'Total_Shipments']
        analysis['Delay_Rate'] = (
            analysis['Delayed_Count'] / analysis['Total_Shipments'] * 100
        ).round(2)
        
        return analysis.sort_values('Avg_Delay', ascending=False)
    
    def search_containers_by_status(self, status: str) -> pd.DataFrame:
        """Search containers by status"""
        if self.shipments is None:
            self.load_all_data()
            
        return self.shipments[
            self.shipments['Status'].str.upper() == status.upper()
        ]

# Singleton instance
_data_loader = None

def get_data_loader(data_dir: str = "data") -> DataLoader:
    """Get or create data loader instance"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader(data_dir)
        _data_loader.load_all_data()
    return _data_loader