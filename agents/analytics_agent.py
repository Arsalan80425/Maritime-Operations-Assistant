"""
Enhanced Analytics Agent with Predictive Features
- Delay prediction using time series analysis
- Anomaly detection
- Advanced pattern recognition
- Improved error handling
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.data_loader import get_data_loader
import pandas as pd
import numpy as np
import json
import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AnalyticsAgent:
    """Enhanced agent with predictive analytics capabilities"""
    
    def __init__(self, model_name: str = "mistral"):
        logger.info("Initializing Enhanced Analytics Agent")
        
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.3,
            base_url="http://localhost:11434"
        )
        self.data_loader = get_data_loader()
        self.vector_store = None
        
        # Analytics prompts
        self.analytics_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a maritime analytics expert with predictive capabilities.
            Analyze shipping data and provide actionable insights.
            Focus on:
            - Delay patterns and trends
            - Port performance metrics
            - Predictive insights
            - Risk identification
            - Anomaly detection
            
            Always support your analysis with specific data points."""),
            ("human", "{query}\n\nData Analysis:\n{data}")
        ])
        
        # RAG-enhanced prompt
        self.rag_analytics_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a maritime analytics expert with access to historical data.
            Analyze shipping data using:
            1. Current statistical data
            2. Historical context from knowledge base
            3. Predictive modeling insights
            
            Provide:
            - Pattern analysis with historical comparison
            - Trend identification and forecasting
            - Predictive insights based on patterns
            - Risk assessment with precedents
            - Anomaly detection
            - Actionable recommendations
            
            Always cite data points and historical examples."""),
            ("human", "{query}\n\nCurrent Data:\n{data}\n\nHistorical Context:\n{context}")
        ])
        
        self.chain = self.analytics_prompt | self.llm | StrOutputParser()
        self.rag_chain = self.rag_analytics_prompt | self.llm | StrOutputParser()
        
        logger.info("✅ Analytics Agent initialized")
    
    def _get_rag_context(self, query: str, k: int = 5) -> str:
        """Get relevant historical context"""
        if not self.vector_store or not self.vector_store.vector_store:
            return "No historical context available"
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            
            if not results:
                return "No relevant historical data found"
            
            context_parts = []
            for i, doc in enumerate(results, 1):
                doc_type = doc.metadata.get('type', 'unknown')
                context_parts.append(f"[Historical Record {i} - {doc_type}]\n{doc.page_content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error retrieving RAG context: {e}")
            return "Error retrieving historical context"
    
    def _use_rag_analysis(self) -> bool:
        """Check if RAG is available"""
        return self.vector_store is not None and self.vector_store.vector_store is not None
    
    def _detect_anomalies(self, data: pd.Series, threshold: float = 2.0) -> Dict:
        """Detect anomalies using statistical methods"""
        try:
            mean = data.mean()
            std = data.std()
            
            # Z-score method
            z_scores = np.abs((data - mean) / std)
            anomalies = data[z_scores > threshold]
            
            return {
                "anomalies_detected": len(anomalies),
                "anomaly_values": anomalies.tolist(),
                "mean": float(mean),
                "std": float(std),
                "threshold_used": threshold
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {"error": str(e)}
    
    def _calculate_moving_average(self, data: pd.Series, window: int = 7) -> pd.Series:
        """Calculate moving average for trend analysis"""
        try:
            return data.rolling(window=window, min_periods=1).mean()
        except Exception as e:
            logger.error(f"Error calculating moving average: {e}")
            return data
    
    def _predict_simple_forecast(self, historical_data: pd.Series, periods: int = 7) -> Dict:
        """Simple time series forecasting using moving average"""
        try:
            # Calculate trend
            recent_trend = historical_data.tail(14).mean()
            older_trend = historical_data.head(14).mean() if len(historical_data) >= 28 else recent_trend
            
            # Calculate growth rate
            if older_trend != 0:
                growth_rate = (recent_trend - older_trend) / older_trend
            else:
                growth_rate = 0
            
            # Generate predictions
            last_value = historical_data.iloc[-1]
            predictions = []
            
            for i in range(1, periods + 1):
                predicted_value = last_value * (1 + growth_rate * i * 0.1)
                predictions.append(float(predicted_value))
            
            return {
                "predictions": predictions,
                "prediction_days": periods,
                "trend": "increasing" if growth_rate > 0 else "decreasing" if growth_rate < 0 else "stable",
                "growth_rate": float(growth_rate * 100),
                "confidence": "medium"  # Simple model = medium confidence
            }
            
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            return {"error": str(e)}
    
    def predict_future_delays(self, port_name: str, days_ahead: int = 7) -> str:
        """NEW: Predict future delays for a port using time series analysis"""
        try:
            logger.info(f"Predicting delays for {port_name} - {days_ahead} days ahead")
            
            # Get historical data
            historical_delays = self.data_loader.get_port_delays(port_name, days=30)
            
            if historical_delays.empty:
                return f"❌ Insufficient data for {port_name} to make predictions"
            
            # Prepare time series
            delays_series = historical_delays.sort_values('Date')['Avg_Delay']
            
            # Calculate statistics
            current_avg = delays_series.tail(7).mean()
            overall_avg = delays_series.mean()
            
            # Detect anomalies
            anomaly_info = self._detect_anomalies(delays_series)
            
            # Calculate moving average
            ma_7 = self._calculate_moving_average(delays_series, window=7)
            
            # Generate forecast
            forecast = self._predict_simple_forecast(delays_series, periods=days_ahead)
            
            # Prepare prediction data
            prediction_data = {
                'port': port_name,
                'prediction_horizon': f'{days_ahead} days',
                'current_7day_avg': round(current_avg, 2),
                'overall_30day_avg': round(overall_avg, 2),
                'predicted_delays': [round(p, 2) for p in forecast.get('predictions', [])],
                'trend': forecast.get('trend', 'unknown'),
                'growth_rate_percent': round(forecast.get('growth_rate', 0), 2),
                'anomalies_detected': anomaly_info.get('anomalies_detected', 0),
                'recent_moving_avg': round(ma_7.iloc[-1], 2) if len(ma_7) > 0 else 0,
                'confidence_level': forecast.get('confidence', 'medium'),
                'recommendation': self._generate_delay_recommendation(forecast, current_avg)
            }
            
            query = f"Generate predictive analysis for {port_name} delays over next {days_ahead} days"
            data = json.dumps(prediction_data, indent=2)
            
            # Use RAG for historical comparison
            if self._use_rag_analysis():
                context = self._get_rag_context(
                    f"{port_name} historical delays predictions trends patterns",
                    k=5
                )
                response = self.rag_chain.invoke({
                    "query": query,
                    "data": data,
                    "context": context
                })
                return f"📈 **Predictive Analytics (RAG-Enhanced):**\n\n{response}"
            else:
                response = self.chain.invoke({
                    "query": query,
                    "data": data
                })
                return f"📈 **Predictive Analytics:**\n\n{response}"
            
        except Exception as e:
            logger.error(f"Error in delay prediction: {e}", exc_info=True)
            return f"❌ Error predicting delays: {str(e)}"
    
    def _generate_delay_recommendation(self, forecast: Dict, current_avg: float) -> str:
        """Generate recommendations based on predictions"""
        try:
            trend = forecast.get('trend', 'stable')
            growth_rate = forecast.get('growth_rate', 0)
            
            if trend == 'increasing' and growth_rate > 5:
                return "⚠️ HIGH ALERT: Delays expected to increase significantly. Consider rerouting or preparing contingency plans."
            elif trend == 'increasing':
                return "⚠️ CAUTION: Moderate increase in delays expected. Monitor closely."
            elif trend == 'decreasing':
                return "✅ POSITIVE: Delays expected to decrease. Good time for operations."
            else:
                return "ℹ️ STABLE: Delays expected to remain stable. Normal operations."
                
        except Exception as e:
            return "ℹ️ Unable to generate recommendation"
    
    def analyze_port_delays(self, port_name: str, days: int = 7) -> str:
        """Enhanced delay analysis with anomaly detection"""
        try:
            logger.info(f"Analyzing delays for {port_name} - last {days} days")
            
            delays = self.data_loader.get_port_delays(port_name, days)
            
            if delays.empty:
                logger.warning(f"No data for port {port_name}")
                return f"❌ No data available for port {port_name}"
            
            # Calculate statistics
            delay_series = delays['Avg_Delay']
            
            # Detect anomalies
            anomaly_info = self._detect_anomalies(delay_series)
            
            # Calculate moving average
            ma = self._calculate_moving_average(delay_series, window=3)
            
            stats = {
                'port': port_name,
                'period_days': days,
                'avg_delay': round(delays['Avg_Delay'].mean(), 2),
                'max_delay': round(delays['Avg_Delay'].max(), 2),
                'min_delay': round(delays['Avg_Delay'].min(), 2),
                'std_delay': round(delays['Avg_Delay'].std(), 2),
                'avg_vessels': int(delays['Vessels_in_Port'].mean()),
                'weather_distribution': delays['Weather'].value_counts().to_dict(),
                'recent_remarks': delays['Remarks'].value_counts().to_dict(),
                'delay_trend': 'increasing' if delays['Avg_Delay'].iloc[0] > delays['Avg_Delay'].iloc[-1] else 'decreasing',
                'anomalies_detected': anomaly_info.get('anomalies_detected', 0),
                'moving_avg_latest': round(ma.iloc[-1], 2) if len(ma) > 0 else 0
            }
            
            query = f"Analyze delay patterns for {port_name} over the last {days} days with anomaly detection"
            data = json.dumps(stats, indent=2)
            
            # Use RAG if available
            if self._use_rag_analysis():
                context = self._get_rag_context(
                    f"{port_name} port delays weather congestion historical patterns",
                    k=5
                )
                response = self.rag_chain.invoke({
                    "query": query,
                    "data": data,
                    "context": context
                })
                return f"📊 **RAG-Enhanced Analysis:**\n\n{response}"
            else:
                response = self.chain.invoke({
                    "query": query,
                    "data": data
                })
                return response
            
        except Exception as e:
            logger.error(f"Error analyzing port delays: {e}", exc_info=True)
            return f"❌ Error analyzing port delays: {str(e)}"
    
    def analyze_cargo_performance(self) -> str:
        """Enhanced cargo analysis with performance scoring"""
        try:
            logger.info("Analyzing cargo performance")
            
            analysis = self.data_loader.get_cargo_type_analysis()
            
            # Add performance score
            analysis['Performance_Score'] = 100 - (analysis['Avg_Delay'] * 2)
            analysis['Performance_Score'] = analysis['Performance_Score'].clip(0, 100).round(2)
            
            # Identify best and worst performers
            best_cargo = analysis['Performance_Score'].idxmax()
            worst_cargo = analysis['Performance_Score'].idxmin()
            
            cargo_stats = analysis.to_dict('index')
            cargo_stats['best_performer'] = best_cargo
            cargo_stats['worst_performer'] = worst_cargo
            
            query = "Analyze performance and delay patterns across cargo types with performance scoring"
            data = json.dumps(cargo_stats, indent=2)
            
            # Use RAG
            if self._use_rag_analysis():
                context = self._get_rag_context(
                    "cargo types performance delays electronics machinery textiles perishables",
                    k=7
                )
                response = self.rag_chain.invoke({
                    "query": query,
                    "data": data,
                    "context": context
                })
                return f"📦 **RAG-Enhanced Cargo Analysis:**\n\n{response}"
            else:
                response = self.chain.invoke({
                    "query": query,
                    "data": data
                })
                return response
            
        except Exception as e:
            logger.error(f"Error analyzing cargo: {e}", exc_info=True)
            return f"❌ Error analyzing cargo performance: {str(e)}"
    
    def compare_ports(self, port1: str, port2: str) -> str:
        """Enhanced port comparison with scoring"""
        try:
            logger.info(f"Comparing ports: {port1} vs {port2}")
            
            stats1 = self.data_loader.get_port_statistics(port1)
            stats2 = self.data_loader.get_port_statistics(port2)
            
            if not stats1 or not stats2:
                return f"❌ Unable to find data for one or both ports"
            
            delays1 = self.data_loader.get_port_delays(port1, days=30)
            delays2 = self.data_loader.get_port_delays(port2, days=30)
            
            # Calculate performance scores
            def calculate_score(delays_df):
                if delays_df.empty:
                    return 0
                avg_delay = delays_df['Avg_Delay'].mean()
                return max(0, 100 - (avg_delay * 3))
            
            score1 = calculate_score(delays1)
            score2 = calculate_score(delays2)
            
            comparison = {
                port1: {
                    'country': stats1.get('Country'),
                    'vessels_in_port': stats1.get('Vessels in Port'),
                    'traffic_category': stats1.get('Traffic Category'),
                    'port_activity_index': stats1.get('Port Activity Index'),
                    'avg_delay_30d': round(delays1['Avg_Delay'].mean(), 2) if not delays1.empty else 'N/A',
                    'max_delay_30d': round(delays1['Avg_Delay'].max(), 2) if not delays1.empty else 'N/A',
                    'weather_patterns': delays1['Weather'].value_counts().to_dict() if not delays1.empty else {},
                    'performance_score': round(score1, 2)
                },
                port2: {
                    'country': stats2.get('Country'),
                    'vessels_in_port': stats2.get('Vessels in Port'),
                    'traffic_category': stats2.get('Traffic Category'),
                    'port_activity_index': stats2.get('Port Activity Index'),
                    'avg_delay_30d': round(delays2['Avg_Delay'].mean(), 2) if not delays2.empty else 'N/A',
                    'max_delay_30d': round(delays2['Avg_Delay'].max(), 2) if not delays2.empty else 'N/A',
                    'weather_patterns': delays2['Weather'].value_counts().to_dict() if not delays2.empty else {},
                    'performance_score': round(score2, 2)
                },
                'winner': port1 if score1 > score2 else port2,
                'score_difference': abs(round(score1 - score2, 2))
            }
            
            query = f"Compare performance between {port1} and {port2} with scoring"
            data = json.dumps(comparison, indent=2)
            
            if self._use_rag_analysis():
                context = self._get_rag_context(
                    f"{port1} {port2} port comparison performance historical trends",
                    k=6
                )
                response = self.rag_chain.invoke({
                    "query": query,
                    "data": data,
                    "context": context
                })
                return f"⚖️ **RAG-Enhanced Port Comparison:**\n\n{response}"
            else:
                response = self.chain.invoke({
                    "query": query,
                    "data": data
                })
                return response
            
        except Exception as e:
            logger.error(f"Error comparing ports: {e}", exc_info=True)
            return f"❌ Error comparing ports: {str(e)}"
    
    def identify_congestion_risk(self, port_name: Optional[str] = None) -> str:
        """Enhanced congestion analysis with risk scoring"""
        try:
            logger.info(f"Identifying congestion risk for {port_name or 'all ports'}")
            
            if port_name:
                stats = self.data_loader.get_port_statistics(port_name)
                delays = self.data_loader.get_port_delays(port_name, days=7)
                
                if not stats or delays.empty:
                    return f"❌ Insufficient data for {port_name}"
                
                # Calculate risk score
                vessels = stats.get('Vessels in Port', 0)
                avg_delay = delays['Avg_Delay'].mean()
                congestion_days = len(delays[delays['Remarks'] == 'Heavy congestion'])
                
                risk_score = min(100, (vessels * 0.5) + (avg_delay * 5) + (congestion_days * 10))
                
                risk_level = (
                    "🔴 CRITICAL" if risk_score > 70 else
                    "🟡 HIGH" if risk_score > 50 else
                    "🟢 MODERATE" if risk_score > 30 else
                    "✅ LOW"
                )
                
                risk_data = {
                    'port': port_name,
                    'vessels_in_port': stats.get('Vessels in Port'),
                    'traffic_category': stats.get('Traffic Category'),
                    'recent_avg_delay': round(avg_delay, 2),
                    'max_recent_delay': round(delays['Avg_Delay'].max(), 2),
                    'congestion_days': congestion_days,
                    'congestion_remarks': delays['Remarks'].value_counts().to_dict(),
                    'weather_conditions': delays['Weather'].value_counts().to_dict(),
                    'risk_score': round(risk_score, 2),
                    'risk_level': risk_level
                }
                
                query = f"Assess congestion risk for {port_name} with risk scoring"
                
                if self._use_rag_analysis():
                    context = self._get_rag_context(
                        f"{port_name} congestion heavy traffic vessel delays historical patterns",
                        k=6
                    )
                    data = json.dumps(risk_data, indent=2)
                    response = self.rag_chain.invoke({
                        "query": query,
                        "data": data,
                        "context": context
                    })
                    return f"⚠️ **RAG-Enhanced Risk Assessment:**\n\n{response}"
            
            else:
                # System-wide analysis
                all_ports = self.data_loader.port_data
                high_traffic = all_ports[
                    all_ports['Traffic Category'] == 'High'
                ].nlargest(10, 'Vessels in Port')
                
                risk_data = high_traffic[[
                    'Port Name', 'Country', 'Vessels in Port',
                    'Traffic Category', 'Port Activity Index'
                ]].to_dict('records')
                
                query = "Identify ports with highest congestion risk across the system"
                
                if self._use_rag_analysis():
                    context = self._get_rag_context(
                        "port congestion high traffic vessel delays system wide risk patterns",
                        k=8
                    )
                    data = json.dumps(risk_data, indent=2)
                    response = self.rag_chain.invoke({
                        "query": query,
                        "data": data,
                        "context": context
                    })
                    return f"🌐 **System-Wide RAG-Enhanced Risk Analysis:**\n\n{response}"
            
            # Fallback
            data = json.dumps(risk_data, indent=2)
            response = self.chain.invoke({
                "query": query,
                "data": data
            })
            return response
            
        except Exception as e:
            logger.error(f"Error identifying congestion risk: {e}", exc_info=True)
            return f"❌ Error identifying congestion risk: {str(e)}"
    
    def get_weather_impact_analysis(self, port_name: str) -> str:
        """Analyze weather impact with correlation analysis"""
        try:
            logger.info(f"Analyzing weather impact at {port_name}")
            
            delays = self.data_loader.get_port_delays(port_name, days=30)
            
            if delays.empty:
                return f"❌ No data available for {port_name}"
            
            # Group by weather
            weather_impact = delays.groupby('Weather').agg({
                'Avg_Delay': ['mean', 'max', 'min', 'std'],
                'Vessels_in_Port': 'mean',
                'Date': 'count'
            }).round(2)
            
            weather_impact.columns = ['Avg_Delay', 'Max_Delay', 'Min_Delay', 'Std_Delay', 'Avg_Vessels', 'Days_Count']
            
            # Calculate impact severity
            overall_avg = delays['Avg_Delay'].mean()
            weather_impact['Impact_Severity'] = ((weather_impact['Avg_Delay'] - overall_avg) / overall_avg * 100).round(2)
            
            weather_stats = weather_impact.to_dict('index')
            
            # Identify worst weather
            worst_weather = weather_impact['Avg_Delay'].idxmax()
            weather_stats['worst_weather_condition'] = worst_weather
            weather_stats['overall_avg_delay'] = round(overall_avg, 2)
            
            query = f"Analyze weather impact on operations at {port_name} with severity scoring"
            data = json.dumps(weather_stats, indent=2)
            
            if self._use_rag_analysis():
                context = self._get_rag_context(
                    f"{port_name} weather impact rain clear cloudy storm delays operations",
                    k=6
                )
                response = self.rag_chain.invoke({
                    "query": query,
                    "data": data,
                    "context": context
                })
                return f"🌤️ **RAG-Enhanced Weather Impact Analysis:**\n\n{response}"
            else:
                response = self.chain.invoke({
                    "query": query,
                    "data": data
                })
                return response
            
        except Exception as e:
            logger.error(f"Error analyzing weather impact: {e}", exc_info=True)
            return f"❌ Error analyzing weather impact: {str(e)}"
    
    def search_analytics_insights(self, query: str) -> str:
        """RAG-powered semantic search for analytics"""
        if not self._use_rag_analysis():
            return "❌ RAG search not available. Please provide a specific analytics query."
        
        try:
            logger.info(f"Performing semantic analytics search: {query[:50]}...")
            
            results = self.vector_store.similarity_search(query, k=10)
            
            if not results:
                return "❌ No relevant analytics data found for your query."
            
            # Filter for analytics-relevant data
            analytics_docs = [
                doc for doc in results 
                if doc.metadata.get('type') in ['daily_report', 'port', 'shipment']
            ]
            
            if not analytics_docs:
                return "❌ No relevant analytics insights found."
            
            # Prepare context
            context_parts = []
            for i, doc in enumerate(analytics_docs[:8], 1):
                doc_type = doc.metadata.get('type', 'unknown')
                context_parts.append(f"[Source {i} - {doc_type}]\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate insights
            analysis_query = f"Based on maritime data, provide insights for: {query}"
            
            response = self.rag_chain.invoke({
                "query": analysis_query,
                "data": "See historical context below",
                "context": context
            })
            
            return f"🔍 **Semantic Analytics Search Results:**\n\n{response}\n\n📚 *Based on {len(analytics_docs)} historical records*"
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}", exc_info=True)
            return f"❌ Error performing semantic analytics search: {str(e)}"
    
    def get_trend_analysis(self, metric: str = "delays", period_days: int = 30) -> str:
        """Enhanced trend analysis with predictions"""
        try:
            logger.info(f"Analyzing {metric} trends over {period_days} days")
            
            daily_data = self.data_loader.daily_report
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=period_days)
            recent_data = daily_data[daily_data['Date'] >= cutoff]
            
            if recent_data.empty:
                return f"❌ No data available for the last {period_days} days"
            
            # Calculate trends
            if metric.lower() == "delays":
                trend_data = recent_data.groupby(recent_data['Date'].dt.date).agg({
                    'Avg_Delay': 'mean',
                    'Vessels_in_Port': 'sum',
                    'Port_Name': 'count'
                }).round(2)
                
                # Detect trend direction
                delay_series = trend_data['Avg_Delay']
                ma = self._calculate_moving_average(delay_series, window=7)
                
                # Predict next week
                forecast = self._predict_simple_forecast(delay_series, periods=7)
                
                trend_stats = {
                    'metric': 'Average Delays',
                    'period_days': period_days,
                    'current_avg': round(trend_data['Avg_Delay'].iloc[-7:].mean(), 2),
                    'previous_avg': round(trend_data['Avg_Delay'].iloc[-14:-7].mean(), 2) if len(trend_data) >= 14 else 0,
                    'overall_trend': forecast.get('trend', 'stable'),
                    'peak_delay_day': str(trend_data['Avg_Delay'].idxmax()),
                    'peak_delay_value': round(trend_data['Avg_Delay'].max(), 2),
                    'moving_avg_latest': round(ma.iloc[-1], 2) if len(ma) > 0 else 0,
                    'predictions_next_7days': forecast.get('predictions', []),
                    'growth_rate_percent': round(forecast.get('growth_rate', 0), 2)
                }
            else:
                trend_stats = {'error': 'Unsupported metric'}
            
            query = f"Analyze {metric} trends over {period_days} days with predictions"
            data = json.dumps(trend_stats, indent=2)
            
            if self._use_rag_analysis():
                context = self._get_rag_context(
                    f"delay trends patterns historical comparison {period_days} days",
                    k=6
                )
                response = self.rag_chain.invoke({
                    "query": query,
                    "data": data,
                    "context": context
                })
                return f"📈 **RAG-Enhanced Trend Analysis:**\n\n{response}"
            else:
                response = self.chain.invoke({
                    "query": query,
                    "data": data
                })
                return response
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}", exc_info=True)
            return f"❌ Error analyzing trends: {str(e)}"

# Singleton instance
_analytics_agent = None

def get_analytics_agent() -> AnalyticsAgent:
    """Get or create analytics agent instance"""
    global _analytics_agent
    if _analytics_agent is None:
        _analytics_agent = AnalyticsAgent()
    return _analytics_agent