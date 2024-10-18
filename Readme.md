# Recommender Systems: Standard and Robust Versions

This repository contains two implementations of recommender systems:
1. A **standard recommender system**.
2. A **robust recommender system** designed to handle noisy or incomplete data.

Additionally, the project includes a script to visualize and compare the performance of the two systems.

## Project Structure

```
.
├── recommender_system.py         # Standard recommender system implementation
├── robust_recommender_system.py  # Robust recommender system implementation
├── visualize_result.py           # Script to visualize and compare results
├── requirements.txt              # List of required Python packages
└── README.md                     # Project documentation
```

## Getting Started

Follow the instructions below to set up the project on your local machine.

### Prerequisites

- Python 3.x installed on your machine.
- Git to clone the repository.

### 1. Clone the repository

First, clone the repository from GitHub:

```bash
git clone https://github.com/Anilmgr/PRT840_Recommender_System.git
cd PRT840_Recommender_System
```

### 2. Create a Virtual Environment

It's recommended to create a virtual environment to manage the project dependencies.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment (on Windows)
.\venv\Scripts\activate

# Activate the virtual environment (on MacOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies

After activating the virtual environment, install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies for running the project.

## Running the Project

Once the environment is set up and dependencies are installed, you can run the following scripts:

### 1. Run the Standard Recommender System

```bash
python recommender_system.py
```

This will run the basic implementation of the recommender system.

### 2. Run the Robust Recommender System

```bash
python robust_recommender_system.py
```

This script will run the robust version of the recommender system, designed to handle noisy or missing data.

### 3. Visualize and Compare the Results

```bash
python visualize_result.py
```

This script will generate visualizations to compare the performance of the standard and robust recommender systems.

## Requirements

All the dependencies are listed in the `requirements.txt` file. These include:

- Libraries for data handling, machine learning algorithms, and visualizations.

To install these dependencies, run:

```bash
pip install -r requirements.txt
```

## How It Works

- **`recommender_system.py`**: Implements a collaborative filtering-based recommender system.
- **`robust_recommender_system.py`**: Enhances the standard system with mechanisms to improve robustness against noisy or incomplete data.
- **`visualize_result.py`**: Generates comparative plots and visual representations of the results from the two recommender systems.

## Contributing

Contributions are welcome! Please create issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---