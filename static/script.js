// Translation API setup
const translateAPIUrl = "https://rapid-translate-multi-traduction.p.rapidapi.com/t";
const translateAPIKey = "164944310amsh42317543591df12p1cdfb4jsn361155ee8346";

// Function to translate text using the API
async function translateText(text, fromLang, toLang) {
    const payload = {
        from: fromLang,
        to: toLang,
        q: text
    };
    const headers = {
        'x-rapidapi-key': translateAPIKey,
        'x-rapidapi-host': 'rapid-translate-multi-traduction.p.rapidapi.com',
        'Content-Type': 'application/json'
    };
    try {
        const response = await fetch(translateAPIUrl, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        return data.translated_text || text; // Adjust based on actual API response format
    } catch (error) {
        console.error('Translation error:', error);
        return text; // Return original text if translation fails
    }
}

// Initialize SpeechRecognition
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.interimResults = false;
recognition.maxAlternatives = 1;
let isVoiceMode = false; 
let selectedLanguage = 'en';  // Default language is English

// Handle the recognition result
recognition.onresult = function(event) {
    const transcript = event.results[0][0].transcript;
    document.getElementById('user-input').value = transcript;
    sendMessage(true);  // Pass voice input flag
};

// Error handling
recognition.onerror = function(event) {
    console.error("Speech recognition error detected: " + event.error);
    alert("Speech recognition error: " + event.error);
    if (isVoiceMode) {
        recognition.start();
    }
};

// Toggle voice mode
document.getElementById('toggle-voice-button').addEventListener('click', () => {
    isVoiceMode = !isVoiceMode;  // Toggle voice mode state
    const buttonText = isVoiceMode ? 'Disable Voice Mode' : 'Enable Voice Mode';
    document.getElementById('toggle-voice-button').innerText = buttonText;

    if (isVoiceMode) {
        // Start listening if switched to voice mode
        recognition.start();
    } else {
        // Stop listening if switched to text mode
        recognition.stop();
    }
});

// Handle language selection
document.getElementById('language-select').addEventListener('change', (event) => {
    selectedLanguage = event.target.value;
});

// Function to handle sending messages
async function sendMessage(isVoiceInput = false) {
    let userInput = document.getElementById('user-input').value.trim();
    if (userInput === '') return;

    // Translate user input to English before sending to the server
    userInput = await translateText(userInput, selectedLanguage, 'en');

    // Display user message in the chat window
    addMessageToChat(userInput, 'user-message');

    // Clear the input field
    document.getElementById('user-input').value = '';

    // Send user query to the Flask backend
    fetch('/process_query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userInput }),
    })
    .then(response => response.json())
    .then(async data => {
        let botResponse = data.response;

        // Translate bot response back to the selected language
        botResponse = await translateText(botResponse, 'en', selectedLanguage);

        // Display the bot response in the chat window
        addMessageToChat(botResponse, 'bot-message');

        // Speak the bot's response only if the input was voice input
        if (isVoiceInput) {
            speak(botResponse, 'US English Female');
        }

        // Restart speech recognition after bot response if in voice mode
        if (isVoiceMode) {
            setTimeout(() => {
                recognition.start();
            }, 1500);  // Add a slight delay before restarting the recognition
        }
    })
    .catch(error => console.error('Error:', error));
}

// Function to add messages to the chat window
function addMessageToChat(message, className) {
    const chatWindow = document.getElementById('chat-window');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    
    // Check if the message contains links and convert them to clickable links
    const linkRegex = /https?:\/\/[^\s]+/g;
    let messageParts = message.split(linkRegex);
    let links = message.match(linkRegex);

    messageParts.forEach((part, index) => {
        messageDiv.appendChild(document.createTextNode(part));
        if (links && links[index]) {
            let link = document.createElement('a');
            link.href = links[index];
            link.target = "_blank";
            link.innerText = links[index];
            messageDiv.appendChild(link);
        }
    });

    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;  // Auto-scroll to the latest message
}

// Function to speak the bot's response
function speak(text, lang = 'US English Female') {
    responsiveVoice.speak(text, lang);
}

// Event listeners for chat functionalities
document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});