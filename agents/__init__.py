import os
import yaml

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.models import ModelFamily

from config import API_KEY_OAI, API_KEY_DS, API_KEY_ANT, API_KEY_ARC, API_KEY_XAI, API_KEY_MS, API_KEY_ALI, API_KEY_GOOGLE
import os

# Initialize the model clients
# OpenAI similar models
model_info = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
    "family": ModelFamily.UNKNOWN, 
    "structured_output": True,
}

OAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
OAI_REASONER_MODELS = ["o1-mini", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
oai_model_clients = {
    key: OpenAIChatCompletionClient(model=key, 
                                    api_key=API_KEY_OAI, 
                                    ) for key in OAI_MODELS
    }
oai_reasoner_clients = {
    key: OpenAIChatCompletionClient(model=key, 
                                    api_key=API_KEY_OAI, 
                                    ) for key in OAI_REASONER_MODELS
    }

if API_KEY_GOOGLE:
    GOOGLE_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    google_model_clients = {
        key: OpenAIChatCompletionClient(model=key, 
                                    api_key=API_KEY_GOOGLE, 
                                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                                    model_info={
                                        "vision": True,
                                        "function_calling": True,
                                        "json_output": True,
                                        "family": ModelFamily.is_gemini(key), 
                                        "structured_output": True,
                                    }
                                    ) for key in GOOGLE_MODELS
    }

if API_KEY_XAI:
    XAI_TEXT_MODELS = ["grok-3-mini", "grok-3-fast", "grok-3"]
    xai_model_clients = {
        key: OpenAIChatCompletionClient(model=key, 
                                    api_key=API_KEY_XAI, 
                                    base_url="https://api.x.ai/v1",
                                    model_info={
                                        "vision": False,
                                        "function_calling": True,
                                        "json_output": True,
                                        "family": ModelFamily.UNKNOWN, 
                                        "structured_output": False,
                                    }
                                    ) for key in XAI_TEXT_MODELS
    }
    XAI_MULTIMODAL_MODELS = ["grok-4-0709", "grok-code-fast-1"]
    xai_model_clients.update({
        key: OpenAIChatCompletionClient(model=key, 
                                    api_key=API_KEY_XAI, 
                                    base_url="https://api.x.ai/v1",
                                    model_info={
                                        "vision": True,
                                        "function_calling": True,
                                        "json_output": True,
                                        "family": ModelFamily.UNKNOWN, 
                                        "structured_output": False,
                                    }
                                    ) for key in XAI_MULTIMODAL_MODELS
    })
    
    
if API_KEY_DS:
    DS_MODELS = ["deepseek-chat", "deepseek-reasoner"]
    ds_model_clients = {
        key: OpenAIChatCompletionClient(
            model=key, 
            api_key=API_KEY_DS, 
            base_url="https://api.deepseek.com", 
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN, 
                "structured_output": False,
            }
            ) for key in DS_MODELS
        }
else:
    ds_model_clients = {}
    
if API_KEY_ALI:
    ALI_MODELS = ["qwen3-max-preview", "qwen-plus", "qwen-flash"]
    ali_model_clients = {
        key: OpenAIChatCompletionClient(
            model=key, 
            api_key=API_KEY_ALI, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN, 
                "structured_output": False,
            }
            ) for key in ALI_MODELS
        }
else:
    ali_model_clients = {}
    
if API_KEY_MS:
    MS_MODELS = ["kimi-k2-turbo-preview"]
    ms_model_clients = {
        key: OpenAIChatCompletionClient(
            model=key, 
            api_key=API_KEY_MS, 
            base_url="https://api.moonshot.cn/v1", 
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN, 
                "structured_output": False,
            }
            ) for key in MS_MODELS
        }
else:
    ms_model_clients = {}
        
if API_KEY_ARC:
    ARC_MODELS = ["doubao-seed-1-6-thinking-250715", "doubao-seed-1-6-250615"]
    arc_model_clients = {
        key: OpenAIChatCompletionClient(
            model=key, 
            api_key=API_KEY_ARC, 
            base_url="https://ark.cn-beijing.volces.com/api/v3", 
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN, 
                "structured_output": False,
            }
            ) for key in ARC_MODELS
        }
else:
    arc_model_clients = {}

if API_KEY_ANT:
    ANT_MODELS = ["claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514", "claude-opus-4-20250514"]
    
    ant_model_clients = {
        key: AnthropicChatCompletionClient(
            model=key, 
            api_key=API_KEY_ANT, 
            ) for key in ANT_MODELS
        }
else:
    ant_model_clients = {}

model_clients = oai_model_clients | ds_model_clients | ali_model_clients | ant_model_clients | arc_model_clients | oai_reasoner_clients | xai_model_clients | ms_model_clients | google_model_clients