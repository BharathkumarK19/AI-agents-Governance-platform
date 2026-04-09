from agents.crewai_runtime import get_crewai_components
from agents.llm import get_crewai_llm
from agents.tools import TavilySearchError, build_source_bundle, search_tavily


def create_research_agent(llm):
    Agent, _, _, _, _ = get_crewai_components()
    return Agent(
        role="Research Analyst",
        goal="Find accurate and relevant information from the internet.",
        backstory=(
            "Expert researcher who gathers reliable and up-to-date data using tools."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_analysis_agent(llm):
    Agent, _, _, _, _ = get_crewai_components()
    return Agent(
        role="Data Analyst",
        goal="Analyze research data and extract meaningful insights.",
        backstory="Skilled at identifying patterns, trends, and key insights.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_summary_agent(llm):
    Agent, _, _, _, _ = get_crewai_components()
    return Agent(
        role="Report Generator",
        goal="Create clear, structured, and presentation-ready reports.",
        backstory="Expert communicator who simplifies complex insights.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_tasks(research_agent, analysis_agent, summary_agent):
    _, _, _, _, Task = get_crewai_components()
    research_task = Task(
        description=(
            "Research the user query '{query}' using the provided live web findings.\n\n"
            "Source material:\n{source_bundle}\n\n"
            "Focus on trends, facts, and real-world data."
        ),
        expected_output="Detailed research report",
        agent=research_agent,
    )
    analysis_task = Task(
        description=(
            "Analyze the research output for '{query}' and extract key insights, "
            "patterns, and conclusions as bullet points."
        ),
        expected_output="Bullet-point insights",
        agent=analysis_agent,
        context=[research_task],
    )
    summary_task = Task(
        description=(
            "Convert the analysis into a final structured report for '{query}' "
            "with headings and bullet points. Make it presentation-ready."
        ),
        expected_output="Clean final report",
        agent=summary_agent,
        context=[analysis_task],
    )
    return research_task, analysis_task, summary_task


def build_research_crew():
    _, Crew, _, Process, _ = get_crewai_components()
    llm = get_crewai_llm()
    research_agent = create_research_agent(llm)
    analysis_agent = create_analysis_agent(llm)
    summary_agent = create_summary_agent(llm)
    research_task, analysis_task, summary_task = create_tasks(
        research_agent, analysis_agent, summary_agent
    )
    return Crew(
        agents=[research_agent, analysis_agent, summary_agent],
        tasks=[research_task, analysis_task, summary_task],
        process=Process.sequential,
        verbose=True,
        tracing=False,
    )


def run_research_crew(query):
    try:
        results = search_tavily(query)
    except TavilySearchError:
        raise
    source_bundle = build_source_bundle(results)
    crew = build_research_crew()
    final_output = crew.kickoff(
        inputs={"query": query, "source_bundle": source_bundle}
    )
    return {
        "query": query,
        "sources": results,
        "source_bundle": source_bundle,
        "final_answer": final_output.raw,
    }


def research_and_answer(query):
    return run_research_crew(query)["final_answer"]
