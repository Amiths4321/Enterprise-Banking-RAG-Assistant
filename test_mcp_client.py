
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            # Discover available MCP tools
            tools = await session.list_tools()

            print("\nAVAILABLE MCP TOOLS")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            # Test EMI calculation
            print("\nTESTING EMI TOOL")

            result = await session.call_tool(
                "calculate_emi",
                {
                    "principal": 500000,
                    "annual_interest_rate": 10,
                    "tenure_years": 5,
                },
            )

            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)

            # Test document metadata
            print("\nTESTING DOCUMENT METADATA")

            result = await session.call_tool(
                "list_banking_documents",
                {},
            )

            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
