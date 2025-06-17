import os
import sys

# import ai library 

import openai

# import api key 

from apikey import openai_api_key

# import the client object from openai and pass it my apikey 

client = openai.OpenAI(api_key=openai_api_key)

# retrieve the contents of the memory.txt file 

def get_memory():
    with open("memory.txt", "r") as f:
        return f.read()
    
def get_summarized_memory(): 
    with open("summarized_txt", "r") as f: 
        return f.read()
    
# function to ask LLM, define the model 

def ask_llm(prompt, model="gpt-4o-mini", memory=None):

    # create the system prompt 
    system_message = "You are a helpful research assistant. Always respond in 3 sentences or less. Keep the conversation concise and chatty. Here are the user's profile characteristics: "
    
    # if there is a memory, append this message to the system prompt (grows as memory increases, hence a later addition of conversation summary memory)
    if memory:
        system_message += f" You have the ability to see the memory of the conversation. Here is the memory: {memory}."
    
    # creates a response using methods inside of the OpenAI client 
    response = client.chat.completions.create(
        model=model,
        messages=[
            # passes a list of objects, which are key-value pairs (dictionaries), including the system message and system prompt
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    # the llm's response is returned
    return response.choices[0].message.content.strip()


# saves the memory of each interaction to a file 

def save_to_memory(prompt, response, model="gpt-4o-mini"):
    with open("memory.txt", "a") as f:
        #it is formatted, as ai: text, new line, and then user: text, new line 
        f.write(f"Model: {model}\nUser: {prompt}\nAI: {response}\n\n")


# summarizes the memory of each interaction in a single sentence to optimize token usage 

import asyncio

def summarize_memory(prompt, model="gpt-4o-mini"):

    # a system prompt that explains to the AI how to summarize the memory with a few-shot example. 

    system_message = """You are chatting with a user. If a specific interaction is extremely interesting, 
    you should summarize it in a way that is easy to understand and remember, in a single sentence. 
    The goal is to minimize your token usage. 
    
    Here's an example of how you should save the memory: 

    Interaction 1: the user said that he enjoys programming and I responded with 3 ways to get better.
    Interaction 2: the user mentioned he likes to read books about systems design and I recommended he take a course at MIT. 
    
    """

    # a response is created by calling the openai client object and chat completions endpoint 
    response = client.chat.completions.create(
        model=model,
        messages=[
            # system prompt and user message are passed 
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )

    # the summarized text is written to the memory with the first completion [0]
    with open ("summarized_memory.txt", "a") as f:
        f.write(f"Model: {model}\nSummary: {response.choices[0].message.content.strip()}\n\n")

    # the summarized response is returned for debugging 
    return response.choices[0].message.content.strip()


# tool functions with basic implementations

def sentiment_analyzer(text):
    """Analyze the sentiment of the given text"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a sentiment analyzer. Analyze the emotional tone of the text and respond with: POSITIVE, NEGATIVE, or NEUTRAL, followed by a brief explanation."},
            {"role": "user", "content": f"Analyze the sentiment of this text: {text}"}
        ]
    )
    return response.choices[0].message.content.strip()

def writing_improver(text): 
    """Improve the given text"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a writing assistant. Improve the given text by making it clearer, more engaging, and better structured. Return only the improved version."},
            {"role": "user", "content": f"Improve this text: {text}"}
        ]
    )
    return response.choices[0].message.content.strip()

def researcher_agent(topic):
    """Research the given topic"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a research assistant. Provide helpful, accurate information about the given topic in 2-3 sentences."},
            {"role": "user", "content": f"Research and provide information about: {topic}"}
        ]
    )
    return response.choices[0].message.content.strip()

# A tool is designed to parse a response. 
def native_parser(interaction):
    """Parse the interaction and execute the appropriate tool"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a helpful assistant that selects the appropriate tool based on user input.

You have these tools available:
- Sentiment Analyzer: Analyzes the emotional tone of text
- Writing Improver: Helps improve and refine written content  
- Researcher Agent: Researches topics and provides information

You must respond with ONLY a JSON object in this exact format:
{
  "tool": "tool_name_here",
  "arguments": "the_text_to_process"
}

Examples:
- If user asks about emotions: {"tool": "Sentiment Analyzer", "arguments": "the text to analyze"}
- If user wants writing help: {"tool": "Writing Improver", "arguments": "the text to improve"}  
- If user asks questions: {"tool": "Researcher Agent", "arguments": "the topic to research"}

Return ONLY the JSON, no other text."""},
            {"role": "user", "content": f"The user said: {interaction}"}
        ]
    )
    
    # Get the response content
    response_content = response.choices[0].message.content.strip()
    
    # Parse the JSON response
    try:
        import json
        parsed_response = json.loads(response_content)
        tool_name = parsed_response.get("tool")
        arguments = parsed_response.get("arguments")
        
        return tool_name, arguments
        
    except json.JSONDecodeError:
        # If JSON parsing fails, return None values
        print(f"Error: Could not parse response as JSON: {response_content}")
        return None, None


# a tool executor that takes the function that was called and executes it 

def tool_executor(tool_selection, arguments):
    """Execute the appropriate tool"""
    if not tool_selection or not arguments:
        return "Error: Invalid tool selection or arguments."
        
    if tool_selection == "Sentiment Analyzer":
        return sentiment_analyzer(arguments)
    elif tool_selection == "Writing Improver":
        return writing_improver(arguments)
    elif tool_selection == "Researcher Agent":
        return researcher_agent(arguments)
    else:
        return f"Error: Unknown tool '{tool_selection}'. Available tools: Sentiment Analyzer, Writing Improver, Researcher Agent."


# Agent scratchpad to track all interactions and reasoning
class AgentScratchpad:
    def __init__(self):

        # This will give the scratchpad a few places to save different types of data in a list format 

        # interaction var
        self.interactions = []
        # current plan var 
        self.current_plan = []
        # completed steps var  
        self.completed_steps = []
        
    # adds an interaction to the scratchpad 
    def add_interaction(self, step_type, content, result=None):
        """Add an interaction to the scratchpad"""

        # an interaction is composed of the step, the type of step, the content, and the result 
        interaction = {
            "step": len(self.interactions) + 1,
            "type": step_type,  # "observation", "thought", "action", "final_answer"
            "content": content,
            "result": result
        }
        # the result is appended to the interaction variable as a dictionary (key-value pairs)
        self.interactions.append(interaction)
        
    def get_scratchpad_summary(self):
        """Get a formatted summary of all interactions"""
        summary = "SCRATCHPAD:\n"
        for interaction in self.interactions:
            summary += f"Step {interaction['step']} - {interaction['type'].upper()}: {interaction['content']}\n"
            if interaction['result']:
                summary += f"RESULT: {interaction['result']}\n"
            summary += "\n"
        return summary
        
    def clear(self):
        """Clear the scratchpad for new conversation"""
        self.interactions = []
        self.current_plan = []
        self.completed_steps = []

# Enhanced parser that can handle planning and final answer detection


# The parser will always look at how the final answer is being produced. We are essentially creating completions. 


def agent_parser(user_input, scratchpad):
    """Parse user input and determine next action using scratchpad context"""
    
    scratchpad_context = scratchpad.get_scratchpad_summary()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are an intelligent agent that can plan and execute tasks using tools.

You have these tools available:
- Sentiment Analyzer: Analyzes emotional tone of text
- Writing Improver: Improves and refines written content  
- Researcher Agent: Researches topics and provides information

Your response must be a JSON object with ONE of these formats:

FOR PLANNING (when you need to think about approach):
{
  "action_type": "plan",
  "thought": "your reasoning about what needs to be done",
  "plan": ["step 1", "step 2", "step 3"]
}

FOR USING A TOOL:
{
  "action_type": "tool",
  "tool": "tool_name_here",
  "arguments": "text_to_process",
  "reasoning": "why you're using this tool"
}

FOR FINAL ANSWER (when you have enough information):
{
  "action_type": "final_answer",
  "answer": "your complete answer to the user"
}
             

Look at the scratchpad to see what's already been done. Don't repeat actions unnecessarily."""},
            {"role": "user", "content": f"User request: {user_input}\n\nCurrent scratchpad:\n{scratchpad_context}"}
        ]
    )
    
    # Parse the JSON response
    try:
        import json
        response_content = response.choices[0].message.content.strip()
        parsed_response = json.loads(response_content)
        return parsed_response
        
    except json.JSONDecodeError as e:
        print(f"Error parsing agent response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return {"action_type": "error", "message": "Failed to parse agent response"}

# Agent execution loop
def run_agent(user_input, scratchpad, max_steps=5):
    """Run the agent loop until final answer or max steps reached"""
    
    # Adds an interaction to the scratchpad with the observation 
    scratchpad.add_interaction("observation", f"User request: {user_input}")
    
    for step in range(max_steps):
        print(f"\n--- Agent Step {step + 1} ---")
        
        # Get next action from the scratchpad
        next_action = agent_parser(user_input, scratchpad)


        # 
        if next_action["action_type"] == "plan":
            # Agent is planning
            thought = next_action.get("thought", "")
            plan = next_action.get("plan", [])
            
            scratchpad.add_interaction("thought", thought)
            scratchpad.add_interaction("plan", f"Plan: {'; '.join(plan)}")
            
            print(f"🤔 THINKING: {thought}")
            print(f"📋 PLAN: {'; '.join(plan)}")
            
        elif next_action["action_type"] == "tool":
            # Agent wants to use a tool
            tool_name = next_action.get("tool")
            arguments = next_action.get("arguments")
            reasoning = next_action.get("reasoning", "")
            
            print(f"🔧 USING TOOL: {tool_name}")
            print(f"💭 REASONING: {reasoning}")
            
            # Execute the tool
            tool_result = tool_executor(tool_name, arguments)
            
            scratchpad.add_interaction("action", f"Used {tool_name} with: {arguments}", tool_result)
            
            print(f"📊 RESULT: {tool_result}")
            
        elif next_action["action_type"] == "final_answer":
            # Agent has final answer
            final_answer = next_action.get("answer", "")
            
            scratchpad.add_interaction("final_answer", final_answer)
            
            print(f"✅ FINAL ANSWER: {final_answer}")
            return final_answer
            
        elif next_action["action_type"] == "error":
            error_msg = next_action.get("message", "Unknown error")
            print(f"❌ ERROR: {error_msg}")
            return f"Sorry, I encountered an error: {error_msg}"
            
        else:
            print(f"❓ UNKNOWN ACTION: {next_action}")
            
    # If we reach max steps without final answer
    return "I've reached my thinking limit. Let me try to give you the best answer I can based on what I've learned so far."

logic = """

1. Enter a loop. 
2. Understand/read the current situation. 
3. Create a plan for how to solve it. 
4. Call a tool. Append the results to the scratchpad. 
5. See if it has generated a final answer. If not, move to the next step in the plan. 
6. Call a tool. Append the results to the scratchpad. If not, move to the next step in the plan. 
7. See if it has generated a final answer. If not, move to the next step in the plan. 


"""




if __name__ == "__main__":

    # Create the agent scratchpad
    scratchpad = AgentScratchpad()
    
    # set the main loop to true so that it can run the program 
    main_app_loop = True 

    # print the welcome message 
    print("AI: Welcome to your intelligent CLI agent! I can analyze sentiment, improve writing, and research topics.")
    print("AI: I'll show you my thinking process as I work through your requests.")

    # start the main loop 
    while main_app_loop: 
        # take in user input
        user_input = input("\nYou: ")

        # if the user types exit, he can leave the chat interface 
        if user_input.lower() == "exit":
            print("See you later! Come back again when you want to get serious!")
            main_app_loop = False 
        else:
            # Clear scratchpad for new conversation
            scratchpad.clear()
            
            # Run the agent with the user input
            final_answer = run_agent(user_input, scratchpad)
            
            # Print the final answer
            print(f"\n🎯 AI: {final_answer}")
            
            # Save this interaction to memory (optional - you can keep this for long-term memory)
            save_to_memory(user_input, final_answer)
            
            # Summarize memory (optional - you can keep this for long-term memory)
            memory_to_summarize = f"User: {user_input}\nAI: {final_answer}"
            summarize_memory(memory_to_summarize)
