
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def connect_to_banking_mcp():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            response = await session.list_tools()

            print("Connected to Banking MCP Server")
            print("\nAvailable tools:")

            for tool in response.tools:
                print(f"- {tool.name}: {tool.description}")

            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in response.tools
            ]


if __name__ == "__main__":
    tools = asyncio.run(connect_to_banking_mcp())
    print(f"\nDiscovered {len(tools)} tools.")
