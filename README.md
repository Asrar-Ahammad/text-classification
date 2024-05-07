# Text Guardian

This is a Flask-based web application that allows users to classify text as either AI-generated or human-written. The application currently supports English and Hindi languages, with plans to add support for Tamil and more languages in the future.

## Features

- Text classification using a BERT-based multilingual model
- Support for English and Hindi languages
- User-friendly web interface built with Flask

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/Asrar-Ahammad/text-classification
2. Navigate to the project directory:
    ```
    cd text-classification
3. Create and activate a virtual environment (optional but recommended):
    ```
    conda create -p venv
    conda activate venv
4. Install the required dependencies:
    ```
    pip install -r requirements.txt

## Usage

1. Run the Flask application:
    ```
    flask run
2. Open your web browser and visit `http://localhost:5000` to access the AI Text Classifier.

3. Enter the text you want to classify in the provided input field.

4. Select the language (English or Hindi) from the dropdown menu.

5. Click the "Classify" button to obtain the classification result (AI-generated or Human-written).

## Contributing

We welcome contributions to improve the AI Text Classifier and add support for more languages. If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive commit messages
4. Push your changes to your forked repository
5. Submit a pull request to the main repository

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The BERT-based multilingual text classification model was trained using the [HuggingFace Transformers](https://huggingface.co/transformers/) library.
- The web interface was built using the [Flask](https://flask.palletsprojects.com/) micro web framework.