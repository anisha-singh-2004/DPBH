function displayResult(data) {
    var resultElement = document.getElementById("result");
    resultElement.textContent = data.prediction === 1 ? 'Confirm Shaming Detected' : 'Not Confirm Shaming';
}
document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('processButton').addEventListener('click', processText);
});

function processText() {
    var sentence = document.getElementById("textInput").value;
    console.log("Sentence: " + sentence);
    var apiUrl = "http://localhost:5000/predict";  // Change to your Flask server URL
    var requestData = { text: sentence };

    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
    })
    .then(response => response.json())
    .then(data => {
        // Display the result on the page
        displayResult(data);
    })
    .catch(error => {
        console.log("Error: " + error);
    });
}

function displayResult(data) {
    var resultElement = document.getElementById("result");
    resultElement.textContent = data.prediction === 1 ? 'Confirm Shaming Detected' : 'Not Confirm Shaming';
}
