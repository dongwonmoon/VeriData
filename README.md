# Veri-Data: Automated Data Validation and Documentation

Veri-Data is a command-line tool that automates the process of data validation and documentation. It connects to your data source, profiles your data, suggests validation rules, and then validates your data against those rules. It can also generate documentation for your data automatically.

## Features

*   **Data Profiling:** Understand the characteristics of your data, including data types, distributions, and more.
*   **Automated Rule Suggestion:** Leverage the power of Large Language Models (LLMs) like Gemma or GPT to automatically suggest validation rules.
*   **Data Validation:** Validate your data using Great Expectations, a powerful open-source data validation library.
*   **Automated Documentation:** Generate documentation for your data columns automatically.
*   **Extensible:** Easily extend the tool to support new data sources, profilers, suggesters, and validators.

## How it Works

1.  **Configuration:** Configure your data source, components, and target columns in the `config.yml` file.
2.  **Execution:** Run the `cli.py` script with either the `run` or `document` command.
3.  **Data Loading:** The application loads data from your specified data source.
4.  **Profiling:** The data is profiled to gather statistics and metadata.
5.  **Suggestion:** Based on the profile, an LLM suggests validation rules or documentation.
6.  **Validation:** The validator component uses the suggested rules to validate the data and generates a report.
7.  **Documentation:** The `document` command generates documentation for your specified columns.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Pip

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/veri-data.git
    cd veri-data
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  Copy the `config.yml.example` to `config.yml`:
    ```bash
    cp config.yml.example config.yml
    ```
2.  Edit the `config.yml` file to configure your data source, components, and columns to validate.

### Usage

#### Run Validation

To run the data validation pipeline, use the `run` command:

```bash
python cli.py run --config config.yml
```

#### Generate Documentation

To generate documentation for your data, use the `document` command:

```bash
python cli.py document --config config.yml
```

