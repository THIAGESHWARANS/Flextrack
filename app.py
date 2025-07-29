from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

# Function to run the selected exercise script
def run_script(script_name):
    try:
        process = subprocess.Popen(['python', f'scripts/{script_name}.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process.pid
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_exercise', methods=['POST'])
def start_exercise():
    data = request.json
    exercise = data.get('exercise')

    script_map = {
        "Curl Counter": "curl_counter",
        "Squat Counter": "squat_counter",
        "Push-up Counter": "pushup_counter",
        "Jumping Jack Counter": "jumping_jacks_counter",
        "Lunge Counter": "lunges_counter",
        "Plank Tracker": "plank_tracker"
    }

    if exercise in script_map:
        pid = run_script(script_map[exercise])
        return jsonify({"message": f"{exercise} started!", "pid": pid})
    else:
        return jsonify({"error": "Invalid exercise selection"}), 400

if __name__ == '__main__':
    app.run(debug=True)
