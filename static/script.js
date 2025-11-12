document.getElementById('toggle-extra').addEventListener('click', () => {
  const extra = document.getElementById('extra');
  extra.classList.toggle('hidden');
  document.getElementById('toggle-extra').textContent = extra.classList.contains('hidden') ? 'Mostrar más variables' : 'Ocultar variables';
});

document.getElementById('btn-clear').addEventListener('click', () => {
  document.getElementById('form').reset();
  const res = document.getElementById('result');
  res.classList.add('hidden');
});

document.getElementById('btn-predict').addEventListener('click', async () => {
  // --- Validación de rangos ---
  const fields = [
    { id: 'age', min: 0, max: 120, name: 'Edad' },
    { id: 'body_temperature', min: 35, max: 42, name: 'Temperatura (°C)' },
    { id: 'hemoglobin', min: 5, max: 20, name: 'Hemoglobina (g/dL)' },
    { id: 'hospitalization_days', min: 0, max: 365, name: 'Días de hospitalización' },
    { id: 'red_blood_cells', min: 2, max: 7, name: 'Glóbulos rojos (millones/µL)' },
    { id: 'neutrophils', min: 0, max: 100, name: 'Neutrófilos (%)' },
    { id: 'eosinophils', min: 0, max: 100, name: 'Eosinófilos (%)' },
    { id: 'basophils', min: 0, max: 100, name: 'Basófilos (%)' },
    { id: 'monocytes', min: 0, max: 100, name: 'Monocitos (%)' },
    { id: 'lymphocytes', min: 0, max: 100, name: 'Linfocitos (%)' },
    { id: 'AST', min: 5, max: 40, name: 'AST (SGOT) (U/L)' },
    { id: 'ALT', min: 5, max: 45, name: 'ALT (SGPT) (U/L)' },
    { id: 'ALP', min: 30, max: 150, name: 'Fosfatasa alcalina (ALP)' },
    { id: 'total_bilirubin', min: 0.1, max: 2.0, name: 'Bilirrubina total (mg/dL)' },
    { id: 'direct_bilirubin', min: 0.0, max: 0.5, name: 'Bilirrubina directa (mg/dL)' },
    { id: 'indirect_bilirubin', min: 0.0, max: 1.5, name: 'Bilirrubina indirecta (mg/dL)' },
    { id: 'total_proteins', min: 5.5, max: 8.5, name: 'Proteínas totales (g/dL)' },
    { id: 'albumin', min: 3.0, max: 5.0, name: 'Albúmina (g/dL)' },
    { id: 'platelets', min: 10000, max: 45000, name: 'Plaquetas (cél/µL)' },
    { id: 'white_blood_cells', min: 4000, max: 11000, name: 'Leucocitos (cél/µL)' },
    { id: 'hematocrit', min: 0, max: 100, name: 'Hematocrito (%)' },
    { id: 'creatinine', min: 0.5, max: 1.5, name: 'Creatinina (mg/dL)' },
    { id: 'urea', min: 15, max: 50, name: 'Urea (mg/dL)' }
  ];

  for (const f of fields) {
    const el = document.getElementById(f.id);
    if (!el) continue; // Si el input no existe en la página, lo ignora
    const val = parseFloat(el.value);
    if (isNaN(val)) {
      alert(`Por favor ingresa un valor para ${f.name}.`);
      el.focus();
      return;
    }
    if (val < f.min || val > f.max) {
      alert(`${f.name} debe estar entre ${f.min} y ${f.max}.`);
      el.focus();
      return;
    }
  }

  // --- Payload y fetch ---
  const model = document.querySelector('input[name="model"]:checked').value;
  const payload = {
    model: model,
    age: document.getElementById('age').value || null,
    sex: document.getElementById('sex').value || '',
    body_temperature: document.getElementById('body_temperature').value || null,
    hemoglobin: document.getElementById('hemoglobin').value || null,
    headache: document.getElementById('headache').value || '',
    nausea: document.getElementById('nausea').value || ''
  };

  if (!document.getElementById('extra').classList.contains('hidden')) {
    payload.platelets = document.getElementById('platelets').value || null;
    payload.white_blood_cells = document.getElementById('white_blood_cells').value || null;
    payload.hematocrit = document.getElementById('hematocrit').value || null;
    payload.creatinine = document.getElementById('creatinine').value || null;
    payload.urea = document.getElementById('urea').value || null;
    payload.jaundice = document.getElementById('jaundice').value || '';
    payload.rash = document.getElementById('rash').value || '';
    payload.weakness = document.getElementById('weakness').value || '';
  }

  if (!payload.age || !payload.sex || !payload.body_temperature || !payload.hemoglobin) {
    alert('Por favor completa los campos principales: Edad, Sexo, Temperatura y Hemoglobina.');
    return;
  }

  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  btn.textContent = 'Analizando...';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(txt || 'Error en el servidor');
    }

    const json = await resp.json();
    document.getElementById('res-model').textContent = json.model;
    document.getElementById('res-diagnosis').textContent = json.diagnosis;
    document.getElementById('res-confidence').textContent = json.confidence;

    const resInputs = document.getElementById('res-inputs');
    resInputs.innerHTML = `<strong>Datos ingresados:</strong>
      <ul>
        <li>Edad: ${payload.age}</li>
        <li>Sexo: ${payload.sex}</li>
        <li>Temperatura: ${payload.body_temperature} °C</li>
        <li>Hemoglobina: ${payload.hemoglobin} g/dL</li>
        <li>Dolor de cabeza: ${payload.headache || '-'}</li>
        <li>Náuseas: ${payload.nausea || '-'}</li>
        ${payload.platelets ? `<li>Plaquetas: ${payload.platelets}</li>` : ''}
      </ul>`;

    document.getElementById('result').classList.remove('hidden');
    window.scrollTo({ top: document.getElementById('result').offsetTop - 20, behavior: 'smooth' });
  } catch (err) {
    alert('Error: ' + (err.message || err));
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Realizar Diagnóstico';
  }
});
