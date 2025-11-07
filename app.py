
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)
CORS(app)

# --- Modelo difuso ---
ingresos = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'ingresos')
historial = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'historial')
riesgo = ctrl.Consequent(np.arange(0, 101, 1), 'riesgo')

ingresos['bajo'] = fuzz.trapmf(ingresos.universe, [0, 0, 2, 4])
ingresos['medio'] = fuzz.trapmf(ingresos.universe, [3, 5, 6, 8])
ingresos['alto'] = fuzz.trapmf(ingresos.universe, [7, 9, 10, 10])

historial['malo'] = fuzz.trapmf(historial.universe, [0, 0, 2, 4])
historial['regular'] = fuzz.trapmf(historial.universe, [3, 5, 6, 8])
historial['bueno'] = fuzz.trapmf(historial.universe, [7, 9, 10, 10])

riesgo['muy_alto_r'] = fuzz.trapmf(riesgo.universe, [0, 0, 15, 30])
riesgo['alto_r'] = fuzz.trapmf(riesgo.universe, [20, 35, 45, 60])
riesgo['medio_r'] = fuzz.trapmf(riesgo.universe, [50, 60, 70, 80])
riesgo['bajo_r'] = fuzz.trapmf(riesgo.universe, [70, 85, 100, 100])

rules = [
    ctrl.Rule(ingresos['bajo'] & historial['malo'], riesgo['muy_alto_r']),
    ctrl.Rule(ingresos['bajo'] & historial['regular'], riesgo['alto_r']),
    ctrl.Rule(ingresos['bajo'] & historial['bueno'], riesgo['medio_r']),
    ctrl.Rule(ingresos['medio'] & historial['malo'], riesgo['alto_r']),
    ctrl.Rule(ingresos['medio'] & historial['regular'], riesgo['medio_r']),
    ctrl.Rule(ingresos['medio'] & historial['bueno'], riesgo['bajo_r']),
    ctrl.Rule(ingresos['alto'] & historial['malo'], riesgo['medio_r']),
    ctrl.Rule(ingresos['alto'] & historial['regular'], riesgo['bajo_r']),
    ctrl.Rule(ingresos['alto'] & historial['bueno'], riesgo['bajo_r'])
]

credit_ctrl = ctrl.ControlSystem(rules)
credit_simulation = ctrl.ControlSystemSimulation(credit_ctrl)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluar', methods=['POST'])
def evaluar():
    data = request.get_json()
    val_ingresos = float(data['ingresos'])
    val_historial = float(data['historial'])

    credit_simulation.input['ingresos'] = val_ingresos
    credit_simulation.input['historial'] = val_historial
    credit_simulation.compute()

    puntaje_riesgo = credit_simulation.output['riesgo']

    if puntaje_riesgo <= 30:
        decision = "RECHAZADO (Riesgo Muy Alto)"
    elif puntaje_riesgo <= 60:
        decision = "REQUIERE REVISIÓN MANUAL"
    elif puntaje_riesgo <= 80:
        decision = "APROBADO - Estándar"
    else:
        decision = "APROBADO - Preferente"

    return jsonify({
        'puntaje_riesgo': round(puntaje_riesgo, 2),
        'decision': decision
    })

if __name__ == '__main__':
    app.run(debug=True)
