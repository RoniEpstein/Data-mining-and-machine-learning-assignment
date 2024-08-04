// Initialize Flatpickr for date fields
flatpickr(".datepicker", {
    dateFormat: "d/m/Y",  // Date format
    locale: "he",  // Set language to Hebrew
    allowInput: true,  // Allow manual input
    defaultDate: null,  // Do not set default date
    onReady: function (selectedDates, dateStr, instance) {
        // Ensure placeholder is shown when no date is selected
        if (!dateStr) {
            instance.input.setAttribute('placeholder', 'dd/mm/yyyy');  // Set placeholder
        }
    }
});

// Define options for select fields
const manufacturers = ['אאודי', 'אברת\'', 'אוטוביאנקי', 'אולדסמוביל', 'אוסטין', 'אופל', 'אינפיניטי', 'אלפא רומיאו', 'אם. ג\'י. / MG', 'ב.מ.וו', 'ביואיק', 'גרייט וול / G.O', 'דאצ\'יה', 'דודג\'', 'דייהו', 'דייהטסו', 'הונדה', 'וולוו', 'טויוטה', 'טסלה', 'יגואר', 'יונדאי', 'לאדה', 'לינקולן', 'לנצ\'יה', 'לקסוס', 'מאזדה', 'מזראטי', 'מיני', 'מיצובישי', 'מרצדס', 'ניסאן', 'סאאב', 'סאנגיונג', 'סובארו', 'סוזוקי', 'סיאט', 'סיטרואן', 'סמארט', 'סקודה', 'פולקסווגן', 'פונטיאק', 'פורד', 'פורשה', 'פיאט', 'פיג\'ו', 'פרארי', 'קאדילק', 'קיה', 'קרייזלר', 'רובר', 'רנו', 'שברולט'];
// List of car manufacturers

const currentYear = new Date().getFullYear();  // Get the current year
const years = Array.from({ length: currentYear - 1989 }, (_, i) => currentYear - i);  // Generate years from current year to 1990

const hands = Array.from({ length: 10 }, (_, i) => i + 1);  // Generate numbers 1 to 10

const engineVolumes = [800, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3500, 3600, 3800, 4000, 4200, 4500, 4700, 4900, 5000, 5500, 6000];
// List of engine volumes

const testYears = Array.from({ length: 7 }, (_, i) => 2020 + i);  // Generate years from 2020 to 2026

const colors = ['אדום', 'אדום מטאלי', 'אפור', 'אפור מטאלי', 'אפור עכבר', 'בורדו', 'בורדו מטאלי', 'בז\'', 'בז\' מטאלי', 'ברונזה', 'ברונזה מטאלי', 'ורוד', 'זהב', 'זהב מטאלי', 'חום', 'חום מטאלי', 'חרדל', 'טורקיז', 'ירוק', 'ירוק בהיר', 'ירוק בקבוק', 'ירוק מטאלי', 'כחול', 'כחול בהיר', 'כחול בהיר מטאלי', 'כחול כהה', 'כחול כהה מטאלי', 'כחול מטאלי', 'כסוף', 'כסוף מטאלי', 'כסף מטלי', 'כתום', 'לבן', 'לבן מטאלי', 'לבן פנינה', 'לבן שנהב', 'סגול', 'סגול חציל', 'סגול מטאלי', 'צהוב', 'צהוב מטאלי', 'שחור', 'שמפניה', 'תכלת', 'תכלת מטאלי'];
// List of colors

document.addEventListener('DOMContentLoaded', function () {
    // Populate select fields immediately
    populateSelect('Input_Manufacturer', manufacturers);
    populateSelect('Input_ModelYear', years);
    populateSelect('Input_Hand', hands);
    populateSelect('Input_EngineVolume', engineVolumes);
    populateSelect('Input_TestYear', testYears);
    populateSelect('Input_TestMonth', Array.from({ length: 12 }, (_, i) => i + 1));
    populateSelect('Input_Color', colors);

    // Set placeholders for date inputs
    document.getElementById('Input_CreDate').setAttribute('placeholder', 'בחר תאריך');
    document.getElementById('Input_RepubDate').setAttribute('placeholder', 'בחר תאריך');

    // Limit number of pictures to positive values only
    const picNumInput = document.getElementById('Input_PicNum');
    picNumInput.addEventListener('input', function () {
        let value = parseInt(this.value);
        if (isNaN(value) || value < 0) {
            this.value = '';
        } else {
            this.value = value;
        }
    });

    // List of all form fields
    const allFields = [
        'Input_Manufacturer', 'Input_Model', 'Input_ModelYear', 'Input_Hand',
        'Input_TransType', 'Input_EngineType', 'Input_EngineVolume', 'Input_Km',
        'Input_TestMonth', 'Input_TestYear', 'Input_Color', 'Input_PrevHolder',
        'Input_CurrentHolder', 'Input_Description', 'Input_PicNum', 'Input_Area',
        'Input_City', 'Input_CreDate', 'Input_RepubDate'];

    // Add event listeners to hide result when fields change
    allFields.forEach(fieldId => {
        const element = document.getElementById(fieldId);
        if (element) {
            element.addEventListener('change', hideResult);
            if (element.tagName === 'INPUT' && element.type === 'text') {
                element.addEventListener('input', hideResult);
            }
        }
    });

    // Form submission handling
    document.getElementById('carForm').onsubmit = async (e) => {
        e.preventDefault();  // Prevent form default submission

        const form = e.target;
        const formData = new FormData(form);

        // Check optional fields
        const optionalFields = [
            'Input_Model', 'Input_TestMonth', 'Input_TestYear', 'Input_Description',
            'Input_EngineType', 'Input_EngineVolume', 'Input_Km', 'Input_PrevHolder',
            'Input_CurrentHolder', 'Input_PicNum', 'Input_Area', 'Input_City',
            'Input_CreDate', 'Input_RepubDate', 'Input_Color'];

        const emptyOptionalFields = optionalFields.filter(field => !formData.get(field) || formData.get(field) === '');

        if (emptyOptionalFields.length > 0) {
            const userChoice = await showConfirmationDialog(emptyOptionalFields);
            if (!userChoice) {
                return;  // User chose to go back and fill the fields
            }
        }

        // Fill empty fields with 'None'
        allFields.forEach(field => {
            if (!formData.get(field) || formData.get(field) === '') {
                formData.set(field, 'None');
            }
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.text();
            document.getElementById('result').innerHTML = `<div class="price-label">המחיר המוערך</div><div class="price-value">${result} ₪</div>`;
            document.getElementById('result').style.display = 'block';

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while trying to predict the price. Please try again.');
        }
    };
});

function hideResult() {
    const resultElement = document.getElementById('result');
    resultElement.style.display = 'none';
    resultElement.innerText = '';
}

function showConfirmationDialog(emptyFields) {
    return new Promise((resolve) => {
        const fieldNames = {
            'Input_Model': 'דגם',
            'Input_TestMonth': 'חודש טסט',
            'Input_TestYear': 'שנת טסט',
            'Input_Color': 'צבע',
            'Input_Description': 'תיאור',
            'Input_EngineType': 'סוג מנוע',
            'Input_EngineVolume': 'נפח מנוע',
            'Input_Km': 'ק"מ',
            'Input_PrevHolder': 'בעלות קודמת',
            'Input_CurrentHolder': 'בעלות נוכחית',
            'Input_PicNum': 'מספר תמונות',
            'Input_Area': 'אזור',
            'Input_City': 'עיר',
            'Input_CreDate': 'תאריך העלאת מודעה',
            'Input_RepubDate': 'תאריך עדכון'
        };

        const emptyFieldsNames = emptyFields.map(field => fieldNames[field]).join(', ');

        const dialog = document.createElement('div');
        dialog.innerHTML = `
            <div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center;">
                <div style="background: white; padding: 20px; border-radius: 5px; text-align: center;">
                    <p>The fields:</p>
                    <p>| ${emptyFieldsNames} |</p>
                    <p>are left empty</p>
                    <p>Do you want to continue anyway?</p>
                    <button id="continueBtn" style="margin-right: 10px;">Continue</button>
                    <button id="cancelBtn">Go Back</button>
                </div>
            </div>
        `;

        document.body.appendChild(dialog);

        document.getElementById('continueBtn').onclick = () => {
            document.body.removeChild(dialog);
            resolve(true);  // User chose to continue

        };

        document.getElementById('cancelBtn').onclick = () => {
            document.body.removeChild(dialog);
            resolve(false);  // User chose to go back
        };
    });
}

function populateSelect(elementId, options) {
    const select = document.getElementById(elementId);
    if (!select) return;  // Exit if the element doesn't exist

    // Keep the first option (placeholder) and remove any existing options
    const placeholder = select.firstElementChild;
    select.innerHTML = '';
    select.appendChild(placeholder);

    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
}
