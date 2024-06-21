---
layout: page
title: Trade-Helper AI Project
description: AI-powered stock analysis and prediction tool
img: assets/img/stock_analysis.jpg
importance: 1
category: work
related_publications: true
---

# Trade-Helper: AI-Powered Stock Analysis and Prediction Tool

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/stock_market_1.webp" title="Stock Market" class="img-fluid rounded z-depth-1" %}
    </div>
</div>



## [**Demo: Visit Trade-Helper on Hugging Face Spaces**](https://huggingface.co/spaces/wangzerui/finaien)


[https://huggingface.co/spaces/wangzerui/finaien](https://huggingface.co/spaces/wangzerui/finaien)


Trade-Helper is a cutting-edge AI-powered tool designed to revolutionize stock analysis and prediction. This project leverages various APIs to gather crucial financial information, conduct detailed analyses, and forecast future stock movements. Here’s a detailed overview of the features and functionalities of Trade-Helper:

### Key Features

- **Real-time Company Information**: Access up-to-date company profiles, including business descriptions, industry classifications, and key executives.
- **Comprehensive Financial Data**: Retrieve essential financial metrics such as revenue, earnings, balance sheets, and cash flow statements.
- **Market News Integration**: Aggregate the latest market news and insights related to specific companies to inform analysis.
- **Stock Price Data**: Obtain and process current and historical stock price data to identify trends and patterns.
- **AI Analysis and Prediction**: Utilize advanced AI models to analyze data and predict stock movements, providing actionable insights for investors.

### Technologies Used

- **Python**: The primary programming language used for development, offering robust data manipulation and analysis capabilities.
- **APIs**: Integration with FinnHub, Yahoo Finance, and other financial data sources to ensure comprehensive and accurate data collection.
- **Gradio**: A user-friendly interface for interactive web-based applications.
- **AutoGen**: Configuration and management of AI agents to automate the analysis process.


## Project Workflow

### Environment Setup

We start by loading environment variables and registering API keys to ensure secure access to various data sources:

### Data Processing

Utility functions handle data saving, loading, and formatting to streamline the analysis process:

```python
import pandas as pd
from datetime import date

def save_output(data: pd.DataFrame, tag: str, save_path: str = None) -> None:
    if save_path:
        data.to_csv(save_path)
        print(f"{tag} saved to {save_path}")

def get_current_date():
    return date.today().strftime("%Y-%m-%d")
```

### AI Configuration

Setting up the AI agents with appropriate configurations to ensure precise and efficient analysis:

```python
import autogen

config_list = [
    {
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
]
llm_config = {"config_list": config_list, "timeout": 120, "temperature": 0}

analyst = autogen.AssistantAgent(
    name="Market_Analyst",
    system_message="As a Market Analyst, ...",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)
```

### Tool Registration

Registering tools for data retrieval and analysis ensures a seamless flow of information:

```python
from finrobot.toolkits import register_toolkits
from finrobot.data_source import FinnHubUtils, YFinanceUtils

tools = [
    {
        "function": FinnHubUtils.get_company_profile,
        "name": "get_company_profile",
        "description": "get a company's profile information"
    },
    {
        "function": FinnHubUtils.get_company_news,
        "name": "get_company_news",
        "description": "retrieve market news related to designated company"
    },
    {
        "function": FinnHubUtils.get_basic_financials,
        "name": "get_financial_basics",
        "description": "get latest financial basics for a designated company"
    },
    {
        "function": YFinanceUtils.get_stock_data,
        "name": "get_stock_data",
        "description": "retrieve stock price data for designated ticker symbol"
    }
]
register_toolkits(tools, analyst, user_proxy)
```

### Interactive Interface

Creating an interactive interface using Gradio, allowing users to easily access and interact with the analysis tool:

```python
import gradio as gr

iface = gr.Interface(
    fn=analyze_company,
    inputs=gr.Textbox(lines=1, placeholder="Enter company name or stock code"),
    outputs=gr.Markdown(label="Trade-Helper"),
    title="Trade-Helper",
    description="Enter the company name or stock code to get a AI-Powered analysis and forecast prediction.",
    css=custom_css,
    allow_flagging='never'
)

if __name__ == "__main__":
    iface.launch(share=True)
```

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/ai_interface.png" title="Trade Helper Interface" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The AI-powered Trade-Helper interface providing real-time stock analysis.
</div>

## Conclusion

Trade-Helper exemplifies the powerful integration of AI technology with financial analysis, offering users precise, data-driven insights and predictions. The project underscores my capabilities in utilizing advanced AI models and integrating multiple data sources to deliver comprehensive market analysis. This tool not only serves as a testament to my technical skills but also highlights my ability to develop practical solutions that can significantly benefit investors and financial analysts.

For more information about this project and my other work, please visit deployed service https://huggingface.co/spaces/wangzerui/finaien. Feel free to reach out for collaboration opportunities or consultation.

