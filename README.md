<h1>WhatsApp Persona Mirror</h1>
<p><strong>WhatsApp Persona Mirror</strong> is a Streamlit application that analyzes WhatsApp chat history from JSON exports, generating personalized responses that imitate a selected participant's communication style and tone. Users can interact with the app to explore different conversational styles based on previous chats.</p>

<h2>Features</h2>
<ul>
    <li><strong>Role-Based Imitation</strong>: Select a chat participant’s role from the WhatsApp JSON file to generate responses in their unique style.</li>
    <li><strong>Tone Customization</strong>: Choose from multiple conversation tones—Charming and Fun, Sarcastic, or Professional—to add personality to responses.</li>
    <li><strong>Context-Aware Responses</strong>: Integrates relevant chat context using FAISS vector stores for more accurate, contextual replies.</li>
    <li><strong>Streamlined Interface</strong>: Upload chat files, select parameters, and receive style-adapted responses on the go.</li>
</ul>

<h2>Installation</h2>
<ol>
    <li><strong>Clone the repository</strong>:
        <pre><code>(https://github.com/siddheshsonawane07/mimic_user_persona_using_mistral)
cd whatsapp-persona-mirror
        </code></pre>
    </li>
    <li><strong>Set up a virtual environment</strong>:
        <pre><code>python3 -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
        </code></pre>
    </li>
    <li><strong>Install dependencies</strong>:
        <pre><code>pip install -r requirements.txt
        </code></pre>
    </li>
    <li><strong>Environment Variables</strong>:<br>
        Create a <code>.env</code> file in the project root with your API key:
        <pre><code>MISTRAL_API_KEY=your_mistral_api_key
        </code></pre>
    </li>
    <li><strong>Run the Streamlit app</strong>:
        <pre><code>streamlit run app.py
        </code></pre>
    </li>
</ol>

<h2>Usage</h2>
<ol>
    <li><strong>Upload a WhatsApp Chat JSON File</strong>:
        <ul>
            <li>Export your chat from WhatsApp.</li>
            <li>Upload this file on to this website to get the required format <code>https://sidjsonifywhatsapptext.streamlit.app/</code>.</li>
        </ul>
    </li>
    <li><strong>Select Role and Tone</strong>:
        <ul>
            <li>Choose a participant's role from the dropdown list.</li>
            <li>Pick a conversation tone (Charming and Fun, Sarcastic, Professional).</li>
        </ul>
    </li>
    <li><strong>Start Chatting</strong>:
        <ul>
            <li>Type a message and observe responses tailored to your chosen style and tone.</li>
            <li>Clear the chat history anytime using the <strong>Clear Chat</strong> button.</li>
        </ul>
    </li>
</ol>

<h2>Project Structure</h2>
<pre><code>
├── app.py                    # Main Streamlit application
├── README.md                 # Project documentation
└── .env.example              # Sample environment file
</code></pre>

<h2>Dependencies</h2>
<ul>
    <li><a href="https://streamlit.io/">Streamlit</a> for app interface</li>
    <li><a href="https://github.com/facebookresearch/faiss">FAISS</a> for vector-based similarity search</li>
    <li><a href="https://www.mistral.ai/">Mistral AI</a> for embedding and chat models</li>
    <li><a href="https://pypi.org/project/tiktoken/">tiktoken</a> for token estimation</li>
</ul>

<h2>Troubleshooting</h2>
<ul>
    <li><strong>Environment Variable Errors</strong>: Ensure <code>MISTRAL_API_KEY</code> is correctly set in your <code>.env</code> file.</li>
    <li><strong>JSON Upload Errors</strong>: Only valid JSON files exported from WhatsApp are supported. If there’s an error, try re-exporting the JSON file.</li>
</ul>
