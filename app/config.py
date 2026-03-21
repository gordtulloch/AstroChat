from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llama_server_url: str = "http://127.0.0.1:8081"
    mcp_server_url: str = "https://soleilesb.sunrisesd.ca:8001"

    app_host: str = "localhost"
    app_port: int = 8080

    default_system_prompt: str = "You are a helpful system prompt for Sunrise School Division staff. Follow instructions" \
                                " carefully. Be concise. If you don't know the answer, say you don't know." \
                                " Always be polite and professional. Use Canadian English." \
                                " Inform the user that this is a prototype AI system, " \
                                "and may not always provide accurate information. Do not mention that you are running locally or " \
                                "on a user's machine. If asked about your identity, say you are an AI assistant for Sunrise School Division. " \
                                "If asked about your capabilities, say you can assist with a variety of tasks and answer " \
                                "questions to the best of your ability, but right now you don't have access to any student, staff, or other " \
                                "useful data but stay tuned as that is the eventual aim." 
    default_max_tokens: int = 1024
    default_context_size: int = 4096

    # ChromaDB / RAG settings
    chroma_db_path: str = "data/chromadb"
    chroma_collection: str = "documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 3
    rag_enabled: bool = True

    # Microsoft Entra authentication
    entra_auth_enabled: bool = False
    entra_tenant_id: str = ""
    entra_spa_client_id: str = ""
    entra_api_client_id: str = ""
    entra_api_scope: str = ""
    entra_redirect_uri: str = ""


settings = Settings()
