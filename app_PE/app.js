const STORAGE_KEY = "planning_app_2026_v1";

function loadStore() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || { year: 2026, doctors: {} };
  } catch {
    return { year: 2026, doctors: {} };
  }
}

function saveStore(store) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

// Helper: récupère "M" ou "A" depuis "Matin/Après-midi"
function slotCode(label) {
  return label === "Matin" ? "M" : "A";
}

// Récupère les données d’une fiche médecin (1 section)
function readDoctorSection(sectionEl) {
  const doctorName = sectionEl.dataset.doctor;

  // nb jours/semaine
  const maxJoursInput = sectionEl.querySelector('input[type="number"]');
  const max_jours = maxJoursInput ? parseInt(maxJoursInput.value || "0", 10) : 0;

  // salles autorisées (checkboxes sous "Salles autorisées")
  const salles_autorisees = Array.from(
    sectionEl.querySelectorAll('div b:contains("Salles autorisées")')
  );

  // Comme :contains n’existe pas en CSS natif, on prend une méthode simple:
  // on prend TOUTES les checkboxes qui ont un label "Salle ..." ou "Imagerie/Consultation"
  const roomChecks = Array.from(sectionEl.querySelectorAll('input[type="checkbox"]'))
    .filter(cb => {
      const txt = cb.parentElement?.textContent?.trim() || "";
      return ["Salle 1","Salle 2","Salle 3","Imagerie","Consultation"].includes(txt);
    });

  const rooms = roomChecks.filter(cb => cb.checked).map(cb => cb.parentElement.textContent.trim());

  // jours interdits = ce que le médecin DÉCOCHE dans disponibilités
  // On lit les blocs "Lundi..Samedi" puis Matin/Après-midi
  // Ici on interprète : si non coché => interdit
  const jours = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi"];
  const jours_interdits = [];

  jours.forEach(day => {
    const dayCard = Array.from(sectionEl.querySelectorAll(".mini"))
      .find(div => div.querySelector("h3")?.textContent?.trim() === day);

    if (!dayCard) return;

    const checks = Array.from(dayCard.querySelectorAll('input[type="checkbox"]'));
    // On suppose ordre: Matin puis Après-midi
    if (checks[0] && !checks[0].checked) jours_interdits.push([day, "M"]);
    if (checks[1] && !checks[1].checked) jours_interdits.push([day, "A"]);
  });

  // vacances: on récupère toutes les dates sélectionnées (start/end)
  const dateInputs = Array.from(sectionEl.querySelectorAll('input[type="date"]'));
  // On stocke les dates telles quelles (YYYY-MM-DD) ; ton backend décidera comment les traiter
  const vacances = dateInputs.map(i => i.value).filter(Boolean);

  // notes
  const notes = sectionEl.querySelector("textarea")?.value || "";

  return {
    name: doctorName,
    max_jours,
    salles_autorisees: rooms,
    jours_interdits,
    vacances,
    notes
  };
}

// Remplit une fiche médecin depuis le store (optionnel mais pratique)
function fillDoctorSection(sectionEl, data) {
  const maxJoursInput = sectionEl.querySelector('input[type="number"]');
  if (maxJoursInput) maxJoursInput.value = data.max_jours ?? 0;

  // salles autorisées
  const roomChecks = Array.from(sectionEl.querySelectorAll('input[type="checkbox"]'))
    .filter(cb => {
      const txt = cb.parentElement?.textContent?.trim() || "";
      return ["Salle 1","Salle 2","Salle 3","Imagerie","Consultation"].includes(txt);
    });

  roomChecks.forEach(cb => {
    const room = cb.parentElement.textContent.trim();
    cb.checked = (data.salles_autorisees || []).includes(room);
  });

  // disponibilités: si [jour,plage] est dans jours_interdits => décocher
  const forbidden = new Set((data.jours_interdits || []).map(x => `${x[0]}_${x[1]}`));
  const jours = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi"];
  jours.forEach(day => {
    const dayCard = Array.from(sectionEl.querySelectorAll(".mini"))
      .find(div => div.querySelector("h3")?.textContent?.trim() === day);
    if (!dayCard) return;

    const checks = Array.from(dayCard.querySelectorAll('input[type="checkbox"]'));
    if (checks[0]) checks[0].checked = !forbidden.has(`${day}_M`);
    if (checks[1]) checks[1].checked = !forbidden.has(`${day}_A`);
  });

  // vacances: on ne remplit pas automatiquement (car tu as des paires start/end + ajout dynamique)
  if (sectionEl.querySelector("textarea")) sectionEl.querySelector("textarea").value = data.notes || "";
}

// Sauvegarde au clic sur "Enregistrer"
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".save-doctor");
  if (!btn) return;

  const section = btn.closest("section[id^='doc-']");
  if (!section) return;

  const store = loadStore();
  const data = readDoctorSection(section);
  store.doctors[data.name] = data;
  saveStore(store);

  alert(`✅ Données enregistrées pour ${data.name}`);
});

// Au chargement : re-remplir les fiches depuis le store
document.addEventListener("DOMContentLoaded", () => {
  const store = loadStore();
  document.querySelectorAll("section[id^='doc-'][data-doctor]").forEach(section => {
    const name = section.dataset.doctor;
    if (store.doctors[name]) {
      fillDoctorSection(section, store.doctors[name]);
    }
  });
});

// Bouton "Optimiser" (envoie au backend)
async function runOptimization() {
  const store = loadStore();

  // Convertit store -> payload pour ton solver
  const payload = {
    year: 2026,
    doctors: Object.values(store.doctors),
    // à compléter avec tes salles + disponibilités_salles depuis ta page "planning des salles"
    salles: [
        "mammographie 1",
        "echographie 2",
        "echographie 3",
        "scanner",
        "IRM",
        "radiographie Saint Sever",
      ],
    disponibilites_salles: {
      "Salle 1": ["Lundi M", "Lundi A"],
      "Salle 2": ["Mardi M"],
    },
    block_weeks: 8,
    max_time: 5,
    alpha: 100,
    beta: 1,
  };

  const res = await fetch("http://127.0.0.1:8000/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const out = await res.json();
  console.log(out);
  alert(`Résultat: ${out.status} | assignments: ${(out.assignments || []).length}`);
}

// si tu veux un bouton : <button onclick="runOptimization()">Optimiser</button>
window.runOptimization = runOptimization;