import asyncio
import os
import sys
import json
import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"  # You can change this to your preferred model

class OllamaMCPClient:
    def __init__(self, mcp_server_script="mcp_server.py", ollama_url=OLLAMA_BASE_URL):
        self.mcp_server_script = mcp_server_script
        self.ollama_url = ollama_url
        
    async def get_memory(self):
        """Get conversation memory from file"""
        try:
            with open("memory.txt", "r") as f:
                return f.read()
        except FileNotFoundError:
            return ""
    
    async def save_to_memory(self, prompt, response, model=DEFAULT_MODEL):
        """Save interaction to memory file"""
        with open("memory.txt", "a") as f:
            f.write(f"Model: {model}\nUser: {prompt}\nAI: {response}\n\n")
    
    async def ask_ollama(self, prompt, model=DEFAULT_MODEL, memory=None):
        """Ask Ollama model with optional memory context"""
        system_message = "You are a helpful research assistant. Always respond in 3 sentences or less. Keep the conversation concise and chatty."
        
        if memory:
            system_message += f" You have the ability to see the memory of the conversation. Here is the memory: {memory}"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_url}/api/chat", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['message']['content'].strip()
                    else:
                        return f"Error: Ollama API returned status {response.status}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    async def summarize_memory_ollama(self, prompt, model=DEFAULT_MODEL):
        """Summarize memory using Ollama model"""
        system_message = """You are chatting with a user. If a specific interaction is extremely interesting, 
        you should summarize it in a way that is easy to understand and remember, in a single sentence. 
        The goal is to minimize your token usage. 
        
        Here's an example: 
        The user said that he enjoys programming and I responded with 3 ways to get better.
        """
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_url}/api/chat", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary = result['message']['content'].strip()
                        
                        # Save summary to file
                        with open("summarized_memory.txt", "a") as f:
                            f.write(f"Model: {model}\nSummary: {summary}\n\n")
                        
                        return summary
                    else:
                        return f"Error: Ollama API returned status {response.status}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    async def get_mcp_enhanced_response(self, user_input, user_id="user123"):
        """Get response enhanced with MCP data using stdio transport"""
        try:
            # Set up server parameters for MCP connection
            server_params = StdioServerParameters(
                command="python3",
                args=[self.mcp_server_script],
                env=None
            )
            
            # Connect to MCP server using stdio
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the MCP session
                    await session.initialize()
                    
                    # List available tools (optional - for debugging)
                    tools_response = await session.list_tools()
                    print(f"Available MCP tools: {[tool.name for tool in tools_response.tools]}")
                    
                    # Get user profile and memory using actual MCP tools
                    try:
                        # Get user profile
                        profile_result = await session.call_tool(
                            "get_user_profile_characteristics", 
                            arguments={"user_id": user_id}
                        )
                        profile = str(profile_result.content[0].text) if profile_result.content else "No profile data"
                        
                        # Get MCP memory
                        memory_result = await session.call_tool(
                            "get_user_memory",
                            arguments={"user_id": user_id}
                        )
                        mcp_memory = str(memory_result.content[0].text) if memory_result.content else "No MCP memory"
                        
                        # Use researcher agent for initial processing
                        researcher_result = await session.call_tool(
                            "use_researcher_agent",
                            arguments={"user_input": f"Context: {profile}\nMemory: {mcp_memory}\nUser input: {user_input}"}
                        )
                        researcher_response = str(researcher_result.content[0].text) if researcher_result.content else "No researcher response"
                        
                    except Exception as e:
                        print(f"MCP tool call failed: {e}")
                        profile = "Profile data not available"
                        mcp_memory = "MCP memory not available"
                        researcher_response = "Researcher agent not available"
                    
                    # Get local memory
                    local_memory = await self.get_memory()
                    
                    # Combine contexts
                    enhanced_context = f"User Profile: {profile}\nMCP Memory: {mcp_memory}\nLocal Memory: {local_memory}\nResearcher Response: {researcher_response}"
                    
                    # Get final response from Ollama with all context
                    final_prompt = f"Based on this context: {enhanced_context}\nRespond to user: {user_input}"
                    response = await self.ask_ollama(final_prompt, memory=local_memory)
                    
                    return response
                    
        except Exception as e:
            # Fallback to simple Ollama response if MCP fails
            print(f"MCP Error: {e}, falling back to local Ollama...")
            memory = await self.get_memory()
            return await self.ask_ollama(user_input, memory=memory)

async def main():
    # Initialize the enhanced client
    client = OllamaMCPClient()
    
    print("AI: Welcome to your custom CLI with MCP integration and local Ollama!")
    print("AI: Make sure Ollama is running locally (ollama serve)")
    print("AI: Also ensure your MCP server script exists (default: mcp_server.py)")
    
    main_app_loop = True
    
    while main_app_loop:
        try:
            user_input = input("You: ")
            
            if user_input.lower() == "exit":
                main_app_loop = False
                print("AI: Goodbye!")
                break
            
            # Get enhanced response using both MCP and Ollama
            response = await client.get_mcp_enhanced_response(user_input)
            
            # Save to memory
            await client.save_to_memory(user_input, response)
            
            # Summarize for memory optimization
            memory_to_summarize = f"User: {user_input}\nAI: {response}"
            await client.summarize_memory_ollama(memory_to_summarize)
            
            print("AI: " + response)
            
        except KeyboardInterrupt:
            print("\nAI: Goodbye!")
            break
        except Exception as e:
            print(f"AI: Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
