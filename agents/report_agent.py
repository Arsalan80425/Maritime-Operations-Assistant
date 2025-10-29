"""
Report Generator Agent - Enhanced with RAG Support
Creates comprehensive reports and summaries with context from vector store
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.data_loader import get_data_loader
from datetime import datetime, timedelta
import pandas as pd
import json

class ReportAgent:
    """Agent responsible for generating reports with RAG enhancement"""
    
    def __init__(self, model_name: str = "mistral"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            base_url="http://localhost:11434"
        )
        self.data_loader = get_data_loader()
        self.vector_store = None  # Will be set by orchestrator
        
        # Create report prompt template with RAG context
        self.report_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a maritime operations report specialist.
            Generate professional, comprehensive reports with:
            - Executive summary
            - Key metrics and statistics
            - Trend analysis
            - Actionable recommendations
            - Clear formatting with sections
            
            Additional Context (if available):
            {rag_context}
            
            Make reports data-driven and actionable."""),
            ("human", "{query}\n\nReport Data:\n{data}")
        ])
        
        self.chain = self.report_prompt | self.llm | StrOutputParser()
    
    def _get_rag_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context from vector store for reports"""
        if not self.vector_store or not self.vector_store.vector_store:
            return "No additional context available"
        
        try:
            # Search for relevant documents
            results = self.vector_store.similarity_search(query, k=k)
            
            if not results:
                return "No relevant historical data found"
            
            # Filter and format results
            context_parts = []
            
            # Separate by document type for better organization
            ports_data = [doc for doc in results if doc.metadata.get('type') == 'port']
            daily_reports = [doc for doc in results if doc.metadata.get('type') == 'daily_report']
            shipments = [doc for doc in results if doc.metadata.get('type') == 'shipment']
            
            if ports_data:
                context_parts.append("**Port Information:**")
                for doc in ports_data[:2]:
                    port_name = doc.metadata.get('port_name', 'Unknown')
                    context_parts.append(f"- {port_name}: {doc.page_content[:200]}...")
            
            if daily_reports:
                context_parts.append("\n**Recent Operational Data:**")
                for doc in daily_reports[:2]:
                    port = doc.metadata.get('port', 'Unknown')
                    date = doc.metadata.get('date', 'Unknown')
                    context_parts.append(f"- {port} ({date}): {doc.page_content[:200]}...")
            
            if shipments:
                context_parts.append("\n**Related Shipments:**")
                for doc in shipments[:2]:
                    container = doc.metadata.get('container_id', 'Unknown')
                    status = doc.metadata.get('status', 'Unknown')
                    context_parts.append(f"- {container} ({status}): {doc.page_content[:150]}...")
            
            return "\n".join(context_parts) if context_parts else "No relevant context found"
            
        except Exception as e:
            print(f"⚠️  Error retrieving RAG context for report: {e}")
            return f"Context retrieval error: {str(e)}"
    
    def generate_weekly_delay_report(self, port_name: str = None) -> str:
        """Generate weekly delay summary report with RAG enhancement"""
        try:
            # Get RAG context for historical comparison
            if port_name:
                rag_query = f"historical delay data for {port_name} port past weeks trends patterns"
            else:
                rag_query = "system-wide delay patterns historical trends weekly reports"
            
            rag_context = self._get_rag_context(rag_query, k=5)
            
            if port_name:
                # Single port report
                delays = self.data_loader.get_port_delays(port_name, days=7)
                
                if delays.empty:
                    return f"No data available for {port_name}"
                
                report_data = {
                    'report_type': 'Weekly Delay Report',
                    'port': port_name,
                    'period': '7 days',
                    'total_days_recorded': len(delays),
                    'average_delay': round(delays['Avg_Delay'].mean(), 2),
                    'max_delay': round(delays['Avg_Delay'].max(), 2),
                    'min_delay': round(delays['Avg_Delay'].min(), 2),
                    'average_vessels': int(delays['Vessels_in_Port'].mean()),
                    'weather_conditions': delays['Weather'].value_counts().to_dict(),
                    'operational_status': delays['Remarks'].value_counts().to_dict(),
                    'daily_breakdown': delays[['Date', 'Avg_Delay', 'Vessels_in_Port', 'Weather', 'Remarks']].to_dict('records')
                }
            else:
                # System-wide report
                all_delays = self.data_loader.daily_report
                cutoff = datetime.now() - timedelta(days=7)
                week_data = all_delays[all_delays['Date'] >= cutoff]
                
                port_summary = week_data.groupby('Port_Name').agg({
                    'Avg_Delay': ['mean', 'max', 'min'],
                    'Vessels_in_Port': 'mean'
                }).round(2)
                
                report_data = {
                    'report_type': 'System-Wide Weekly Delay Report',
                    'period': '7 days',
                    'total_ports': len(week_data['Port_Name'].unique()),
                    'total_records': len(week_data),
                    'overall_avg_delay': round(week_data['Avg_Delay'].mean(), 2),
                    'ports_with_high_delays': week_data[week_data['Avg_Delay'] > 10]['Port_Name'].value_counts().to_dict(),
                    'top_5_busiest_ports': week_data.nlargest(5, 'Vessels_in_Port')[['Port_Name', 'Vessels_in_Port', 'Avg_Delay']].to_dict('records'),
                    'weather_impact': week_data.groupby('Weather')['Avg_Delay'].mean().to_dict()
                }
            
            query = f"Generate a comprehensive weekly delay report with trend analysis and recommendations"
            data = json.dumps(report_data, indent=2, default=str)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "rag_context": rag_context
            })
            
            # Add RAG attribution if context was used
            if rag_context and "No additional context" not in rag_context:
                response += "\n\n---\n*📚 Report enhanced with historical data from knowledge base*"
            
            return response
            
        except Exception as e:
            return f"Error generating weekly report: {str(e)}"
    
    def generate_shipment_status_report(self) -> str:
        """Generate current shipment status report with RAG insights"""
        try:
            # Get RAG context for shipment trends
            rag_query = "shipment status trends delay patterns cargo performance historical comparison"
            rag_context = self._get_rag_context(rag_query, k=5)
            
            shipments = self.data_loader.shipments
            
            # Status breakdown
            status_counts = shipments['Status'].value_counts().to_dict()
            
            # Cargo type breakdown
            cargo_status = shipments.groupby(['Cargo_Type', 'Status']).size().unstack(fill_value=0).to_dict('index')
            
            # Port breakdown
            top_ports = shipments['Port_Name'].value_counts().head(10).to_dict()
            
            # Delay statistics
            delayed = shipments[shipments['Status'] == 'Delayed']
            
            # Calculate delay trends by cargo type
            cargo_delay_analysis = shipments.groupby('Cargo_Type').agg({
                'Delay_Hours': ['mean', 'max', 'count']
            }).round(2).to_dict()
            
            # Port performance metrics
            port_performance = shipments.groupby('Port_Name').agg({
                'Delay_Hours': 'mean',
                'Container_ID': 'count'
            }).nlargest(10, 'Container_ID').to_dict()
            
            report_data = {
                'report_type': 'Shipment Status Report',
                'timestamp': str(datetime.now()),
                'total_shipments': len(shipments),
                'status_breakdown': status_counts,
                'delayed_shipments': len(delayed),
                'average_delay_hours': round(delayed['Delay_Hours'].mean(), 2) if len(delayed) > 0 else 0,
                'max_delay_hours': round(delayed['Delay_Hours'].max(), 2) if len(delayed) > 0 else 0,
                'cargo_type_status': cargo_status,
                'cargo_delay_analysis': cargo_delay_analysis,
                'top_10_ports': top_ports,
                'port_performance': port_performance,
                'critical_delays': delayed[delayed['Delay_Hours'] > 5].to_dict('records')[:10],
                'on_time_percentage': round((len(shipments[shipments['Delay_Hours'] == 0]) / len(shipments)) * 100, 2)
            }
            
            query = "Generate a comprehensive shipment status report with performance insights and trend analysis"
            data = json.dumps(report_data, indent=2, default=str)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "rag_context": rag_context
            })
            
            # Add RAG attribution
            if rag_context and "No additional context" not in rag_context:
                response += "\n\n---\n*📚 Analysis includes insights from historical shipment patterns*"
            
            return response
            
        except Exception as e:
            return f"Error generating shipment report: {str(e)}"
    
    def generate_port_performance_report(self, port_name: str) -> str:
        """Generate detailed port performance report with RAG context"""
        try:
            # Get RAG context specific to this port
            rag_query = f"{port_name} port performance historical data operational trends congestion patterns"
            rag_context = self._get_rag_context(rag_query, k=7)
            
            # Get port statistics
            port_stats = self.data_loader.get_port_statistics(port_name)
            
            if not port_stats:
                return f"Port {port_name} not found"
            
            # Get operational data
            delays = self.data_loader.get_port_delays(port_name, days=30)
            delays_7d = self.data_loader.get_port_delays(port_name, days=7)
            
            # Get shipments at this port
            shipments = self.data_loader.shipments
            port_shipments = shipments[shipments['Port_Name'].str.upper() == port_name.upper()]
            
            # Calculate trend (7-day vs 30-day average)
            avg_delay_7d = round(delays_7d['Avg_Delay'].mean(), 2) if not delays_7d.empty else 0
            avg_delay_30d = round(delays['Avg_Delay'].mean(), 2) if not delays.empty else 0
            trend = "improving" if avg_delay_7d < avg_delay_30d else "worsening" if avg_delay_7d > avg_delay_30d else "stable"
            
            # Weather impact analysis
            weather_impact = {}
            if not delays.empty:
                weather_impact = delays.groupby('Weather').agg({
                    'Avg_Delay': 'mean',
                    'Date': 'count'
                }).to_dict()
            
            # Peak congestion times
            congestion_days = []
            if not delays.empty:
                congestion_days = delays[delays['Remarks'] == 'Heavy congestion']['Date'].tolist()
            
            report_data = {
                'report_type': 'Port Performance Report',
                'port_name': port_name,
                'country': port_stats.get('Country'),
                'un_code': port_stats.get('UN Code'),
                'port_type': port_stats.get('Type'),
                'static_metrics': {
                    'vessels_in_port': port_stats.get('Vessels in Port'),
                    'traffic_category': port_stats.get('Traffic Category'),
                    'port_activity_index': port_stats.get('Port Activity Index'),
                    'active_ratio': port_stats.get('Active Ratio'),
                    'traffic_density': port_stats.get('Traffic Density')
                },
                'operational_performance_30d': {
                    'avg_delay': avg_delay_30d,
                    'max_delay': round(delays['Avg_Delay'].max(), 2) if not delays.empty else 'N/A',
                    'min_delay': round(delays['Avg_Delay'].min(), 2) if not delays.empty else 'N/A',
                    'avg_vessels': int(delays['Vessels_in_Port'].mean()) if not delays.empty else 'N/A',
                    'weather_distribution': delays['Weather'].value_counts().to_dict() if not delays.empty else {}
                },
                'operational_performance_7d': {
                    'avg_delay': avg_delay_7d,
                    'trend': trend,
                    'congestion_days': len(congestion_days),
                    'avg_vessels': int(delays_7d['Vessels_in_Port'].mean()) if not delays_7d.empty else 'N/A'
                },
                'weather_impact_analysis': weather_impact,
                'shipment_statistics': {
                    'total_shipments': len(port_shipments),
                    'status_breakdown': port_shipments['Status'].value_counts().to_dict(),
                    'cargo_types': port_shipments['Cargo_Type'].value_counts().to_dict(),
                    'avg_delay_per_shipment': round(port_shipments['Delay_Hours'].mean(), 2),
                    'delayed_shipments': len(port_shipments[port_shipments['Status'] == 'Delayed'])
                },
                'efficiency_score': round((100 - avg_delay_30d * 2), 2) if avg_delay_30d < 50 else 0  # Simple efficiency metric
            }
            
            query = f"Generate a comprehensive performance report for {port_name} with insights, trends, and recommendations"
            data = json.dumps(report_data, indent=2, default=str)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "rag_context": rag_context
            })
            
            # Add RAG attribution with specifics
            if rag_context and "No additional context" not in rag_context:
                response += f"\n\n---\n*📚 Report enriched with historical data and operational patterns for {port_name}*"
            
            return response
            
        except Exception as e:
            return f"Error generating port performance report: {str(e)}"
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary of entire operation with RAG insights"""
        try:
            # Get broad RAG context for executive overview
            rag_query = "maritime operations summary key metrics trends performance indicators strategic insights"
            rag_context = self._get_rag_context(rag_query, k=8)
            
            shipments = self.data_loader.shipments
            ports = self.data_loader.port_data
            daily = self.data_loader.daily_report
            
            # Calculate key metrics
            cutoff = datetime.now() - timedelta(days=7)
            recent_daily = daily[daily['Date'] >= cutoff]
            
            cutoff_30 = datetime.now() - timedelta(days=30)
            monthly_daily = daily[daily['Date'] >= cutoff_30]
            
            # Performance trends
            weekly_avg_delay = round(recent_daily['Avg_Delay'].mean(), 2)
            monthly_avg_delay = round(monthly_daily['Avg_Delay'].mean(), 2)
            delay_trend = "improving" if weekly_avg_delay < monthly_avg_delay else "worsening"
            
            # Top performers and concerns
            best_ports = recent_daily.nsmallest(5, 'Avg_Delay')[['Port_Name', 'Avg_Delay']].to_dict('records')
            worst_ports = recent_daily.nlargest(5, 'Avg_Delay')[['Port_Name', 'Avg_Delay']].to_dict('records')
            
            # Cargo analysis
            cargo_performance = shipments.groupby('Cargo_Type').agg({
                'Delay_Hours': 'mean',
                'Container_ID': 'count'
            }).sort_values('Delay_Hours').to_dict()
            
            # Critical issues
            critical_delays = len(shipments[shipments['Delay_Hours'] > 10])
            heavy_congestion_days = len(recent_daily[recent_daily['Remarks'] == 'Heavy congestion'])
            
            summary_data = {
                'report_type': 'Executive Summary',
                'timestamp': str(datetime.now()),
                'reporting_period': '7 days (with 30-day comparison)',
                'fleet_overview': {
                    'total_shipments': len(shipments),
                    'in_transit': len(shipments[shipments['Status'] == 'In Transit']),
                    'arrived': len(shipments[shipments['Status'] == 'Arrived']),
                    'delayed': len(shipments[shipments['Status'] == 'Delayed']),
                    'on_time_rate': round((len(shipments[shipments['Delay_Hours'] == 0]) / len(shipments)) * 100, 2)
                },
                'network_overview': {
                    'total_ports': len(ports),
                    'high_traffic_ports': len(ports[ports['Traffic Category'] == 'High']),
                    'medium_traffic_ports': len(ports[ports['Traffic Category'] == 'Medium']),
                    'active_ports_7d': len(recent_daily['Port_Name'].unique()),
                    'total_vessels_in_system': int(ports['Vessels in Port'].sum())
                },
                'performance_metrics': {
                    'avg_system_delay_7d': weekly_avg_delay,
                    'avg_system_delay_30d': monthly_avg_delay,
                    'delay_trend': delay_trend,
                    'ports_with_congestion': heavy_congestion_days,
                    'critical_delays': critical_delays,
                    'avg_delay_per_shipment': round(shipments['Delay_Hours'].mean(), 2)
                },
                'top_performers': {
                    'best_performing_ports': best_ports,
                    'best_cargo_types': list(cargo_performance.keys())[:3]
                },
                'top_concerns': {
                    'most_delayed_ports': worst_ports,
                    'high_risk_cargo': shipments[shipments['Status'] == 'Delayed']['Cargo_Type'].value_counts().head(3).to_dict(),
                    'weather_related_delays': recent_daily[recent_daily['Weather'].isin(['Stormy', 'Foggy'])]['Port_Name'].value_counts().to_dict()
                },
                'cargo_performance_summary': cargo_performance,
                'strategic_metrics': {
                    'system_efficiency_score': round(100 - (weekly_avg_delay * 3), 2),
                    'port_utilization': round((ports['Vessels in Port'].mean() / ports['Vessels in Port'].max()) * 100, 2),
                    'delay_impact_hours': int(shipments['Delay_Hours'].sum())
                }
            }
            
            query = "Generate an executive summary with strategic insights, trends, and actionable recommendations"
            data = json.dumps(summary_data, indent=2, default=str)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "rag_context": rag_context
            })
            
            # Add RAG attribution
            if rag_context and "No additional context" not in rag_context:
                response += "\n\n---\n*📚 Executive summary enhanced with historical trends and strategic insights from knowledge base*"
            
            return response
            
        except Exception as e:
            return f"Error generating executive summary: {str(e)}"
    
    def generate_custom_report(self, report_type: str, parameters: dict) -> str:
        """Generate custom report with RAG support"""
        try:
            # Build RAG query based on parameters
            rag_query = f"{report_type} {' '.join(str(v) for v in parameters.values())} trends analysis"
            rag_context = self._get_rag_context(rag_query, k=5)
            
            query = f"Generate a custom {report_type} report"
            data = json.dumps(parameters, indent=2, default=str)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "rag_context": rag_context
            })
            
            return response
            
        except Exception as e:
            return f"Error generating custom report: {str(e)}"

# Create singleton instance
_report_agent = None

def get_report_agent() -> ReportAgent:
    """Get or create report agent instance"""
    global _report_agent
    if _report_agent is None:
        _report_agent = ReportAgent()
    return _report_agent