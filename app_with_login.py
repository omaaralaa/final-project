import streamlit as st
import sqlite3
import hashlib
import ollama
from PIL import Image
from classes_and_functions import predict_image,model,transform
import io
# Initialize the database connection
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    # Create the users table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY, 
                        password TEXT)''')
    
    # Create the chat_history table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        chat_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        chat_input TEXT,
                        chat_output TEXT,
                        username TEXT REFERENCES users(username))''')
    
    # Create the image_history table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS image_history (
                        image_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        image BLOB,
                        image_classification_result TEXT, 
                        recommendation TEXT,
                        username TEXT REFERENCES users(username))''')
    
    conn.commit()
    return conn

# Hash the password for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Verify hashed password
def verify_password(hashed_password, user_password):
    return hashed_password == hashlib.sha256(user_password.encode()).hexdigest()

# Register a new user
def register_user(username, password):
    conn = init_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                       (username, hash_password(password)))
        conn.commit()
        st.success("Account created successfully!")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose a different one.")
    finally:
        conn.close()

# Check login credentials
def login_user(username, password):
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result and verify_password(result[0], password):
        return True
    else:
        return False


# Function to save history
def save_chat_history(chat_input, chat_output, username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (chat_input, chat_output, username)
        VALUES (?, ?, ?)
    """, (chat_input, chat_output, username))
    conn.commit()
    conn.close()


def save_image_history(image, image_classification_result, recommendation, username):
    conn= sqlite3.connect("users.db")  # Assuming this function establishes a database connection
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO image_history (image, image_classification_result, recommendation, username)
        VALUES (?, ?, ?, ?)""", (image, image_classification_result, recommendation, username))
    conn.commit()
    conn.close()

# Streamlit UI
st.title(":green[Welcome to Plant Disease Classification & QA Chatbot]")

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    # Selection of login or register
    choice = st.selectbox("Choose an option", ["Login", "Register"])
    
    if choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
 
        
        if st.button("Login"):
            if login_user(username, password):
                st.session_state["authenticated"] = True  # User is now authenticated
                st.success("Login successful!")
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("Invalid username or password")

    elif choice == "Register":
        st.subheader("Create New Account")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if new_password == confirm_password:
                register_user(new_username, new_password)
            else:
                st.error("Passwords do not match.")

# Main app content - Only visible after successful login
if st.session_state["authenticated"]:
    advises = {
    'Tomato___Late_blight': "Remove infected leaves, apply copper-based fungicides, and avoid overhead watering.",
    'Tomato___healthy': "No action needed. Maintain proper care and monitor for any signs of disease.",
    'Grape___healthy': "No action needed. Ensure proper watering and pruning for continued health.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove infected trees, control psyllid insects with insecticides, and use disease-resistant varieties.",
    'Soybean___healthy': "No action needed. Continue monitoring for pests and diseases.",
    'Squash___Powdery_mildew': "Use sulfur or neem oil sprays, remove infected leaves, and ensure proper air circulation.",
    'Potato___healthy': "No action needed. Maintain soil health and proper irrigation.",
    'Corn_(maize)___Northern_Leaf_Blight': "Rotate crops, remove infected leaves, and use resistant seed varieties.",
    'Tomato___Early_blight': "Remove affected leaves, apply copper fungicides, and avoid wetting foliage when watering.",
    'Tomato___Septoria_leaf_spot': "Apply fungicides, remove infected leaves, and improve air circulation.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Apply fungicides, practice crop rotation, and plant resistant varieties.",
    'Strawberry___Leaf_scorch': "Remove infected leaves, ensure proper watering, and apply fungicides if necessary.",
    'Peach___healthy': "No action needed. Monitor for pests and diseases.",
    'Apple___Apple_scab': "Apply fungicides, remove fallen leaves, and choose resistant apple varieties.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Remove infected plants, control whiteflies with insecticides, and use virus-resistant varieties.",
    'Tomato___Bacterial_spot': "Apply copper-based sprays, remove infected leaves, and avoid overhead watering.",
    'Apple___Black_rot': "Prune infected branches, apply fungicides, and remove fallen fruit to prevent spread.",
    'Blueberry___healthy': "No action needed. Maintain good soil conditions and watering practices.",
    'Cherry_(including_sour)___Powdery_mildew': "Use sulfur-based fungicides, remove affected leaves, and improve air circulation.",
    'Peach___Bacterial_spot': "Apply copper sprays, prune infected branches, and ensure proper air circulation.",
    'Apple___Cedar_apple_rust': "Use fungicides, remove nearby juniper trees (alternate host), and plant resistant varieties.",
    'Tomato___Target_Spot': "Apply fungicides, remove infected leaves, and ensure proper spacing for air circulation.",
    'Pepper,_bell___healthy': "No action needed. Maintain proper care and monitor for disease.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Remove infected leaves, apply fungicides, and avoid wet foliage.",
    'Potato___Late_blight': "Destroy infected plants, apply fungicides, and ensure proper drainage.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants, disinfect tools, and control aphids.",
    'Strawberry___healthy': "No action needed. Maintain proper watering and monitor for pests.",
    'Apple___healthy': "No action needed. Ensure regular pruning and disease monitoring.",
    'Grape___Black_rot': "Apply fungicides, remove affected fruit, and ensure good air circulation.",
    'Potato___Early_blight': "Use fungicides, remove affected leaves, and rotate crops yearly.",
    'Cherry_(including_sour)___healthy': "No action needed. Maintain proper pruning and disease prevention measures.",
    'Corn_(maize)___Common_rust_': "Use resistant corn varieties, remove infected leaves, and apply fungicides if severe.",
    'Grape___Esca_(Black_Measles)': "Prune infected vines, remove affected grapes, and avoid excessive watering.",
    'Raspberry___healthy': "No action needed. Maintain proper watering and pruning.",
    'Tomato___Leaf_Mold': "Ensure good ventilation, apply fungicides, and remove affected leaves.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Spray water or neem oil, introduce natural predators (e.g., ladybugs), and avoid over-fertilizing.",
    'Pepper,_bell___Bacterial_spot': "Use copper-based sprays, remove infected leaves, and avoid handling wet plants.",
    'Corn_(maize)___healthy': "No action needed. Monitor for pests and maintain good field hygiene."
    }

    # Image upload section
    st.header("Plant Disease Classification")
    if "processed_image" not in st.session_state:
        st.session_state["processed_image"] = None
    uploaded_file = st.file_uploader("Upload an image of a plant for disease classification:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="Uploaded Image")
    if uploaded_file is not None and st.session_state["processed_image"] != uploaded_file.name:
        st.session_state["processed_image"] = uploaded_file.name 
        # Display the uploaded image
        img = Image.open(uploaded_file)
        uploaded_img=img
        img = transform(img)
        # Make prediction
        predicted_class = predict_image(img, model)
        st.success(f"Classification Result: {predicted_class}")
        st.success(f'Recommended advice: {advises[predicted_class]}')
        uploaded_img.save(io.BytesIO(), format=uploaded_img.format)
        img_binary = io.BytesIO().getvalue()
        save_image_history(img_binary,predicted_class,advises[predicted_class],username=st.session_state['username'])
    # Set the model name
    model = "agri-qa-model-new"

    # Title of the app
    st.header("Plant Care Q&A Chatbot")

    # Function to generate a response using Ollama
    def generate_response(question):
        try:
            # Call the Ollama API
            response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": question}]
            )
            # Extract and return the response content
            return response["message"]["content"]
        except Exception as e:
            return f"An error occurred: {str(e)}"

    # Create a form for user input
    with st.form("my_form"):
       # Text area for user input
        text = st.text_area(
            "Enter your plant care or agriculture-related question and submit:",
       )

        # Submit button
        submitted = st.form_submit_button("Submit")

        # If the form is submitted, generate and display the response
        if submitted:
            with st.spinner("Generating response..."):  # Show a spinner while processing
                response = generate_response(text)
                st.write(response)  # Display the response in an info box
                save_chat_history(text,response,st.session_state['username'])
    
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute('''
        SELECT chat_input,chat_output FROM chat_history 
        WHERE username = ?''', (st.session_state['username'],))
    chat_data=cursor.fetchall()
    
    cursor.execute('''
        SELECT image_classification_result, recommendation FROM image_history 
        WHERE username = ?''', (st.session_state['username'],))
    image_data=cursor.fetchall()    
    conn.close()
    
    if (not chat_data) and (not image_data):
        st.write("No history found for this user.")
    if chat_data:
        st.subheader("Chat History")
        st.table([{"Question": row[0], "Answer": row[1]} for row in chat_data])
    if image_data:
        st.subheader("Classification History")
        st.table([{"Classification Result": row[0], "Recommended Advice": row[1]} for row in image_data])        

    logout= st.button("Logout")
    if logout:
        st.session_state["authenticated"] = False
        st.rerun()