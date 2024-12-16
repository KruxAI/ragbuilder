def check_graph_dependencies():
    """Check if graph dependencies are installed."""
    try:
        import neo4j
        from langchain_community.graphs import Neo4jGraph
        return True
    except ImportError:
        raise ImportError(
            "Graph dependencies not found. Install them with: "
            "uv pip install 'ragbuilder[graph]'"
        )
