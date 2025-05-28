import os
import sys
import signal
import smtplib
import requests 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import imaplib
import email
from email.header import decode_header

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request 
from starlette.routing import Route, Mount 
from starlette.middleware.cors import CORSMiddleware

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport


load_dotenv()

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

GMAIL_MCP_SERVER_PORT = 8001

def signal_handler(sig, frame):
    print(f"Signal {sig} received. Shutting down Gmail MCP server...")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


mcp_fastmcp_instance = FastMCP( 
    name="gmail-mcp",
    timeout=30
)

def send_email(recipient: str, subject: str, body: str, attachment_path: str = None) -> str:
    try:
        if not SMTP_USERNAME or not SMTP_PASSWORD:
            return "Failed to send email: SMTP credentials not configured."
        msg = MIMEMultipart()
        msg["From"] = SMTP_USERNAME
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        if attachment_path:
            if not os.path.exists(attachment_path):
                return f"Failed to send email: Attachment path '{attachment_path}' does not exist."
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
            msg.attach(part)
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, recipient, msg.as_string())
        server.quit()
        return "Email sent successfully."
    except Exception as e:
        print(f"Error in send_email: {e}")
        return f"Failed to send email: {str(e)}"

def download_attachment_from_url(attachment_url: str, attachment_filename: str) -> str:
    temp_dir = "temp_attachments"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, attachment_filename)
    try:
        response = requests.get(attachment_url, timeout=10)
        response.raise_for_status() 
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading attachment from URL {attachment_url}: {e}")
        raise ValueError(f"Failed to download attachment from URL '{attachment_url}': {e}")

def get_pre_staged_attachment(attachment_name: str) -> str:
    attachment_dir = "available_attachments"
    file_path = os.path.join(attachment_dir, attachment_name)
    return file_path if os.path.exists(file_path) else None

@mcp_fastmcp_instance.tool()
def send_email_tool(recipient: str, subject: str, body: str,
                    attachment_path: str = None,
                    attachment_url: str = None,
                    attachment_name: str = None) -> str:
    final_attachment_path_to_use = attachment_path
    if attachment_url and attachment_name:
        try:
            print(f"Downloading attachment from URL: {attachment_url} as {attachment_name}")
            final_attachment_path_to_use = download_attachment_from_url(attachment_url, attachment_name)
        except ValueError as e: 
            return str(e) 
        except Exception as e: 
            return f"Unexpected error during attachment download: {e}"
    elif attachment_name:
        print(f"Looking for pre-staged attachment: {attachment_name}")
        final_attachment_path_to_use = get_pre_staged_attachment(attachment_name)
        if not final_attachment_path_to_use:
            return f"Error: Pre-staged attachment '{attachment_name}' not found."
    
    print(f"Attempting to send email. Recipient: {recipient}, Subject: '{subject}', Attachment: {final_attachment_path_to_use}")
    return send_email(recipient, subject, body, final_attachment_path_to_use)

@mcp_fastmcp_instance.tool()
def fetch_recent_emails(folder: str = "INBOX", limit: int = 10) -> str:
    print(f"Fetching recent emails from folder: {folder}, limit: {limit}")
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        return "Failed to fetch emails: IMAP credentials not configured."
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(SMTP_USERNAME, SMTP_PASSWORD)
        mail.select(folder)
        result, data = mail.search(None, "ALL")
        if result != 'OK' or not data or not data[0]:
            mail.logout()
            return f"No emails found in folder '{folder}' or error searching."
            
        email_ids_bytes = data[0].split()
        latest_email_ids_bytes = email_ids_bytes[-limit:]
        
        emails_data = []
        for email_id_b in reversed(latest_email_ids_bytes):
            result, msg_data_tuple = mail.fetch(email_id_b, "(RFC822)")
            if result != 'OK':
                continue
            raw_email = msg_data_tuple[0][1]
            msg = email.message_from_bytes(raw_email)
            subject_header_parts = decode_header(msg["Subject"])
            subject = ""
            for part_content, charset in subject_header_parts:
                if isinstance(part_content, bytes):
                    try:
                        subject += part_content.decode(charset or "utf-8", errors="replace")
                    except LookupError:
                        subject += part_content.decode("utf-8", errors="replace")
                else:
                    subject += part_content
            from_ = msg.get("From", "Unknown Sender")
            date = msg.get("Date", "Unknown Date")
            emails_data.append({
                "id": email_id_b.decode(),
                "from": from_,
                "subject": subject,
                "date": date
            })
        mail.close()
        mail.logout()
        if not emails_data:
            return f"No emails found in the folder '{folder}' after fetching."
        result_text = f"Recent emails from '{folder}':\n\n"
        for i, email_item in enumerate(emails_data, 1):
            result_text += f"{i}. From: {email_item['from']}\n"
            result_text += f"   Subject: {email_item['subject']}\n"
            result_text += f"   Date: {email_item['date']}\n"
            result_text += f"   ID: {email_item['id']}\n\n"
        return result_text
    except Exception as e:
        print(f"Error in fetch_recent_emails: {e}")
        return f"Failed to fetch emails: {str(e)}"


sse_transport_handler = SseServerTransport("/messages/") 
async def handle_sse_get_connection(request: Request) -> None:
    
    low_level_mcp_server = mcp_fastmcp_instance._mcp_server
    
    print(f"Incoming SSE connection request to /sse from: {request.client}")
    async with sse_transport_handler.connect_sse(
        request.scope,
        request.receive,
        request._send, 
    ) as (reader, writer):
        print(f"SSE channel connected for {request.client}. Starting MCP protocol run.")
        
        await low_level_mcp_server.run(reader, writer, low_level_mcp_server.create_initialization_options())
        print(f"MCP protocol run finished for {request.client}.")


starlette_main_app = Starlette(
    debug=True, 
    routes=[
        Route("/sse", endpoint=handle_sse_get_connection),
        
        Mount("/messages/", app=sse_transport_handler.handle_post_message),
    ],
)

starlette_main_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000", 
        "http://localhost:8000"   
    ],
    allow_credentials=True,       
    allow_methods=["GET", "POST"],
    allow_headers=["*"],          
)

if __name__ == "__main__":
    print(f"Starting Gmail MCP server (Starlette SSE, adk_mcp.md style) on http://localhost:{GMAIL_MCP_SERVER_PORT}")
    print(f"SSE handshake endpoint (GET): http://localhost:{GMAIL_MCP_SERVER_PORT}/sse")
    print(f"MCP messages endpoint (POST): http://localhost:{GMAIL_MCP_SERVER_PORT}/messages/")
    uvicorn.run(starlette_main_app, host="localhost", port=GMAIL_MCP_SERVER_PORT)