"""
Enhanced Communication Agent with Email/SMS Sending
- SMTP email sending capability
- SMS alerts (simulated Twilio)
- Webhook notifications
- Improved error handling
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.data_loader import get_data_loader
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class CommunicationAgent:
    """Enhanced agent with actual sending capabilities"""
    
    def __init__(self, model_name: str = "mistral"):
        logger.info("Initializing Enhanced Communication Agent")
        
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.5,
            base_url="http://localhost:11434"
        )
        self.data_loader = get_data_loader()
        self.vector_store = None
        
        # Email configuration (set these via environment variables)
        self.smtp_enabled = False  # Set to True when configured
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.smtp_username = ""  # Set via env var
        self.smtp_password = ""  # Set via env var
        
        # SMS configuration (Twilio)
        self.sms_enabled = False  # Set to True when configured
        
        self.comm_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional maritime communications specialist.
            Draft clear, professional communications that:
            - Are concise and informative
            - Include all relevant details
            - Maintain a professional but friendly tone
            - Provide actionable information
            - Follow proper business format
            
            Use provided context to enrich communications.
            Always include subject lines and proper formatting."""),
            ("human", "{query}\n\nContext Data:\n{data}\n\nAdditional Context:\n{context}")
        ])
        
        self.chain = self.comm_prompt | self.llm | StrOutputParser()
        
        logger.info("✅ Communication Agent initialized")
    
    def _get_rag_context(self, query: str, k: int = 3) -> str:
        """Get relevant context from vector store"""
        if not self.vector_store or not self.vector_store.vector_store:
            return "No additional context available"
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            
            if not results:
                return "No relevant historical data found"
            
            context_parts = []
            for i, doc in enumerate(results, 1):
                doc_type = doc.metadata.get('type', 'unknown')
                context_parts.append(f"[Reference {i} - {doc_type}]:\n{doc.page_content[:200]}...\n")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return f"Context retrieval error: {str(e)}"
    
    def send_email(self, recipient: str, subject: str, body: str) -> Dict:
        """
        NEW: Actually send email via SMTP
        Returns: Dict with success status and message
        """
        if not self.smtp_enabled:
            logger.warning("Email sending not configured")
            return {
                "success": False,
                "message": "Email sending not configured. Set SMTP credentials to enable.",
                "simulated": True
            }
        
        try:
            logger.info(f"Sending email to {recipient}: {subject}")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = recipient
            msg['Subject'] = subject
            msg['Date'] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
            
            # Attach body
            msg.attach(MIMEText(body, 'plain'))
            
            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"✅ Email sent successfully to {recipient}")
            
            return {
                "success": True,
                "message": f"Email sent successfully to {recipient}",
                "timestamp": datetime.now().isoformat(),
                "simulated": False
            }
            
        except Exception as e:
            logger.error(f"Error sending email: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error sending email: {str(e)}",
                "simulated": False
            }
    
    def send_sms(self, phone_number: str, message: str) -> Dict:
        """
        NEW: Send SMS via Twilio (simulated for now)
        Returns: Dict with success status
        """
        if not self.sms_enabled:
            logger.warning("SMS sending not configured")
            return {
                "success": False,
                "message": "SMS sending not configured. Set Twilio credentials to enable.",
                "simulated": True
            }
        
        try:
            logger.info(f"Sending SMS to {phone_number}: {message[:50]}...")
            
            # TODO: Implement actual Twilio integration
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)
            # message = client.messages.create(
            #     body=message,
            #     from_=twilio_phone,
            #     to=phone_number
            # )
            
            # For now, simulate
            logger.info(f"📱 [SIMULATED] SMS to {phone_number}: {message}")
            
            return {
                "success": True,
                "message": f"SMS sent to {phone_number} (simulated)",
                "timestamp": datetime.now().isoformat(),
                "simulated": True
            }
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error sending SMS: {str(e)}",
                "simulated": True
            }
    
    def send_webhook(self, webhook_url: str, payload: Dict) -> Dict:
        """
        NEW: Send webhook notification
        Returns: Dict with success status
        """
        try:
            import requests
            
            logger.info(f"Sending webhook to {webhook_url}")
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Webhook sent successfully to {webhook_url}")
                return {
                    "success": True,
                    "message": f"Webhook delivered to {webhook_url}",
                    "status_code": response.status_code
                }
            else:
                logger.warning(f"Webhook returned status {response.status_code}")
                return {
                    "success": False,
                    "message": f"Webhook failed with status {response.status_code}",
                    "status_code": response.status_code
                }
                
        except Exception as e:
            logger.error(f"Error sending webhook: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error sending webhook: {str(e)}"
            }
    
    def draft_delay_notification(self, container_id: str, auto_send: bool = False, 
                                  recipient_email: Optional[str] = None) -> str:
        """Enhanced delay notification with auto-send option"""
        try:
            logger.info(f"Drafting delay notification for {container_id}")
            
            container = self.data_loader.get_container_info(container_id)
            
            if not container:
                logger.warning(f"Container {container_id} not found")
                return f"❌ Container {container_id} not found"
            
            if container['Status'] != 'Delayed':
                return f"ℹ️ Container {container_id} is not delayed (Status: {container['Status']})"
            
            # Get RAG context
            rag_query = f"delays at {container['Port_Name']} port similar issues {container['Cargo_Type']}"
            rag_context = self._get_rag_context(rag_query, k=3)
            
            email_data = {
                'container_id': container_id,
                'port': container['Port_Name'],
                'eta': str(container['ETA']),
                'delay_hours': container['Delay_Hours'],
                'cargo_type': container['Cargo_Type'],
                'status': container['Status']
            }
            
            query = f"Draft a professional delay notification email for container {container_id}"
            data = json.dumps(email_data, indent=2)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "context": rag_context
            })
            
            # Add context note
            if "No additional context" not in rag_context:
                response += f"\n\n---\n*📊 Insights from similar delays at {container['Port_Name']}*"
            
            # Auto-send if requested
            if auto_send and recipient_email:
                subject = f"Delay Alert: Container {container_id}"
                send_result = self.send_email(recipient_email, subject, response)
                
                if send_result['success']:
                    response += f"\n\n✅ **Email sent to {recipient_email}**"
                else:
                    response += f"\n\n⚠️ **Email sending failed:** {send_result['message']}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error drafting delay notification: {e}", exc_info=True)
            return f"❌ Error drafting notification: {str(e)}"
    
    def draft_arrival_notification(self, container_id: str, auto_send: bool = False,
                                   recipient_email: Optional[str] = None) -> str:
        """Enhanced arrival notification with auto-send"""
        try:
            logger.info(f"Drafting arrival notification for {container_id}")
            
            container = self.data_loader.get_container_info(container_id)
            
            if not container:
                return f"❌ Container {container_id} not found"
            
            # Get RAG context
            rag_query = f"arrival procedures {container['Port_Name']} port operations {container['Cargo_Type']}"
            rag_context = self._get_rag_context(rag_query, k=2)
            
            email_data = {
                'container_id': container_id,
                'port': container['Port_Name'],
                'arrival_time': str(container['ETA']),
                'cargo_type': container['Cargo_Type'],
                'status': container['Status']
            }
            
            query = f"Draft a professional arrival notification for container {container_id}"
            data = json.dumps(email_data, indent=2)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "context": rag_context
            })
            
            if "No additional context" not in rag_context:
                response += f"\n\n---\n*📋 Includes operational guidelines from {container['Port_Name']}*"
            
            # Auto-send
            if auto_send and recipient_email:
                subject = f"Arrival Notification: Container {container_id}"
                send_result = self.send_email(recipient_email, subject, response)
                
                if send_result['success']:
                    response += f"\n\n✅ **Email sent to {recipient_email}**"
                else:
                    response += f"\n\n⚠️ **Email sending failed:** {send_result['message']}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error drafting arrival notification: {e}", exc_info=True)
            return f"❌ Error drafting notification: {str(e)}"
    
    def draft_port_congestion_alert(self, port_name: str, auto_send: bool = False,
                                    recipient_email: Optional[str] = None) -> str:
        """Enhanced congestion alert with auto-send"""
        try:
            logger.info(f"Drafting congestion alert for {port_name}")
            
            delays = self.data_loader.get_port_delays(port_name, days=3)
            port_stats = self.data_loader.get_port_statistics(port_name)
            
            if delays.empty or not port_stats:
                return f"❌ Insufficient data for {port_name}"
            
            # Get RAG context
            rag_query = f"congestion patterns {port_name} heavy traffic weather delays historical trends"
            rag_context = self._get_rag_context(rag_query, k=5)
            
            recent_congestion = delays[delays['Remarks'] == 'Heavy congestion']
            avg_delay = delays['Avg_Delay'].mean()
            
            alert_data = {
                'port_name': port_name,
                'country': port_stats.get('Country'),
                'current_vessels': port_stats.get('Vessels in Port'),
                'avg_delay_72h': round(avg_delay, 2),
                'congestion_days': len(recent_congestion),
                'traffic_category': port_stats.get('Traffic Category'),
                'weather_conditions': delays['Weather'].value_counts().to_dict()
            }
            
            query = f"Draft a port congestion alert for {port_name}"
            data = json.dumps(alert_data, indent=2)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "context": rag_context
            })
            
            if "No additional context" not in rag_context:
                response += f"\n\n---\n📊 *Analysis based on historical congestion patterns at {port_name}*"
            
            # Auto-send
            if auto_send and recipient_email:
                subject = f"🚨 Congestion Alert: {port_name}"
                send_result = self.send_email(recipient_email, subject, response)
                
                if send_result['success']:
                    response += f"\n\n✅ **Alert sent to {recipient_email}**"
                else:
                    response += f"\n\n⚠️ **Alert sending failed:** {send_result['message']}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error drafting congestion alert: {e}", exc_info=True)
            return f"❌ Error drafting alert: {str(e)}"
    
    def draft_weekly_status_update(self, recipient_type: str = "client",
                                   auto_send: bool = False,
                                   recipient_email: Optional[str] = None) -> str:
        """Enhanced weekly update with auto-send"""
        try:
            logger.info(f"Drafting weekly status update for {recipient_type}")
            
            shipments = self.data_loader.shipments
            
            # Get RAG context
            rag_query = "weekly shipment trends port performance delays cargo distribution"
            rag_context = self._get_rag_context(rag_query, k=5)
            
            status_counts = shipments['Status'].value_counts().to_dict()
            delayed_count = len(shipments[shipments['Status'] == 'Delayed'])
            avg_delay = round(shipments[shipments['Status'] == 'Delayed']['Delay_Hours'].mean(), 2)
            
            update_data = {
                'recipient_type': recipient_type,
                'total_shipments': len(shipments),
                'status_summary': status_counts,
                'delayed_shipments': delayed_count,
                'average_delay': avg_delay,
                'top_ports': shipments['Port_Name'].value_counts().head(5).to_dict(),
                'cargo_distribution': shipments['Cargo_Type'].value_counts().to_dict()
            }
            
            query = f"Draft weekly status update for {recipient_type}"
            data = json.dumps(update_data, indent=2)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "context": rag_context
            })
            
            if "No additional context" not in rag_context:
                response += "\n\n---\n📈 *Trends powered by historical operational data*"
            
            # Auto-send
            if auto_send and recipient_email:
                subject = f"Weekly Status Update - {datetime.now().strftime('%Y-%m-%d')}"
                send_result = self.send_email(recipient_email, subject, response)
                
                if send_result['success']:
                    response += f"\n\n✅ **Update sent to {recipient_email}**"
                else:
                    response += f"\n\n⚠️ **Update sending failed:** {send_result['message']}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error drafting weekly update: {e}", exc_info=True)
            return f"❌ Error drafting update: {str(e)}"
    
    def generate_sms_alert(self, container_id: str, alert_type: str = "delay",
                          auto_send: bool = False, phone_number: Optional[str] = None) -> str:
        """Enhanced SMS alert with auto-send"""
        try:
            logger.info(f"Generating SMS alert for {container_id}: {alert_type}")
            
            container = self.data_loader.get_container_info(container_id)
            
            if not container:
                return f"❌ Container {container_id} not found"
            
            if alert_type == "delay":
                sms = f"ALERT: Container {container_id} delayed {container['Delay_Hours']}hrs at {container['Port_Name']}. New ETA: {str(container['ETA'])[:10]}"
            elif alert_type == "arrival":
                sms = f"UPDATE: Container {container_id} arrived at {container['Port_Name']}. Ready for pickup."
            else:
                sms = f"STATUS: Container {container_id} - {container['Status']} at {container['Port_Name']}"
            
            sms = sms[:160]  # SMS limit
            
            # Auto-send
            if auto_send and phone_number:
                send_result = self.send_sms(phone_number, sms)
                
                if send_result['success']:
                    sms += f"\n\n✅ **SMS sent to {phone_number}** (simulated)"
                else:
                    sms += f"\n\n⚠️ **SMS sending failed:** {send_result['message']}"
            
            return sms
            
        except Exception as e:
            logger.error(f"Error generating SMS: {e}", exc_info=True)
            return f"❌ Error generating SMS: {str(e)}"
    
    def draft_custom_communication(self, purpose: str, context_query: str,
                                   auto_send: bool = False,
                                   recipient_email: Optional[str] = None,
                                   **kwargs) -> str:
        """Flexible custom communication with auto-send"""
        try:
            logger.info(f"Drafting custom communication: {purpose}")
            
            rag_context = self._get_rag_context(context_query, k=4)
            
            query = f"Draft professional maritime communication for: {purpose}"
            data = json.dumps(kwargs, indent=2, default=str)
            
            response = self.chain.invoke({
                "query": query,
                "data": data,
                "context": rag_context
            })
            
            # Auto-send
            if auto_send and recipient_email:
                subject = purpose
                send_result = self.send_email(recipient_email, subject, response)
                
                if send_result['success']:
                    response += f"\n\n✅ **Communication sent to {recipient_email}**"
                else:
                    response += f"\n\n⚠️ **Sending failed:** {send_result['message']}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error drafting custom communication: {e}", exc_info=True)
            return f"❌ Error drafting communication: {str(e)}"
    
    def search_similar_communications(self, query: str, k: int = 3) -> str:
        """Search for similar past communications"""
        if not self.vector_store or not self.vector_store.vector_store:
            return "❌ RAG search not available"
        
        try:
            logger.info(f"Searching similar communications: {query[:50]}...")
            
            results = self.vector_store.similarity_search(query, k=k)
            
            if not results:
                return "❌ No similar communications found"
            
            similar_cases = []
            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                similar_cases.append({
                    'reference': i,
                    'type': metadata.get('type', 'unknown'),
                    'content': doc.page_content[:200],
                    'metadata': {k: v for k, v in metadata.items() if k != 'type'}
                })
            
            query_prompt = f"Analyze similar cases and provide recommendations: {query}"
            data = json.dumps(similar_cases, indent=2)
            
            response = self.chain.invoke({
                "query": query_prompt,
                "data": data,
                "context": "Historical reference cases"
            })
            
            return f"🔍 **Similar Cases Found:**\n\n{response}"
            
        except Exception as e:
            logger.error(f"Error searching communications: {e}", exc_info=True)
            return f"❌ Error searching: {str(e)}"
    
    def configure_smtp(self, server: str, port: int, username: str, password: str):
        """Configure SMTP settings for email sending"""
        self.smtp_server = server
        self.smtp_port = port
        self.smtp_username = username
        self.smtp_password = password
        self.smtp_enabled = True
        logger.info(f"✅ SMTP configured: {server}:{port}")
    
    def configure_sms(self, account_sid: str, auth_token: str, phone_number: str):
        """Configure Twilio for SMS sending"""
        # Store credentials
        self.sms_enabled = True
        logger.info("✅ SMS (Twilio) configured")

# Singleton
_communication_agent = None

def get_communication_agent() -> CommunicationAgent:
    """Get or create communication agent"""
    global _communication_agent
    if _communication_agent is None:
        _communication_agent = CommunicationAgent()
    return _communication_agent