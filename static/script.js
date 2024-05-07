function countWords() {
  const textarea = document.getElementById("textarea");
  const wordCount = document.getElementById("wordCount");
  const words = textarea.value
    .trim()
    .split(/\s+/)
    .filter((word) => word !== "");
  const numWords = words.length;

  if (numWords > 1000) {
    textarea.value = words.slice(0, 1000).join(" ");
    wordCount.textContent = "1000/1000";
  } else {
    wordCount.textContent = numWords + "/1000";
  }
}

// Function to update placeholder based on selected language
function updatePlaceholder() {
  // Get the selected language from the dropdown
  var languageSelect = document.getElementById("language");
  var selectedLanguage =
    languageSelect.options[languageSelect.selectedIndex].value;

  // Get the textarea element
  var textarea = document.getElementById("textarea");

  // Update the placeholder based on the selected language
  switch (selectedLanguage) {
    case "english":
      textarea.placeholder = "Enter text in English";
      break;
    case "hindi":
      textarea.placeholder = "हिंदी में टेक्स्ट दर्ज करें";
      break;
    case "tamil":
      textarea.placeholder = "தமிழில் உரை உள்ளிடவும்";
      break;
    default:
      textarea.placeholder = "Enter text";
  }
}

// Add event listener to the language select element
document
  .getElementById("language")
  .addEventListener("change", updatePlaceholder);

function chart() {
  const chartText = document.querySelector(".chart-text");
  const chart = document.querySelector(".chart");
  const percen = document.querySelector(".chart-text");

  let inner_text = percen.innerText;
  const regex = /\d+(\.\d+)?/;
  // Simulate data update (replace with your data source)
  let percentage = parseFloat(inner_text.match(regex)[0]);
  console.log(percentage);

  chart.style.width = `${(percentage / 100) * 200}px`;
  chart.style.height = `${(percentage / 100) * 200}px`;

  chartText.textContent = percentage + "%";
}

chart();

// Menu Toggle
const menu_icon = document.querySelector(".menu-icon");
const menu = document.querySelector(".menu-res");

menu_icon.addEventListener("click", () => {
  // Toggle the display property of the menu when the menu icon is clicked
  if (menu.style.display === "flex") {
    // If the menu is currently visible, hide it
    menu.style.display = "none";
  } else {
    // If the menu is currently hidden, show it
    menu.style.display = "flex";
  }
});

function chart2() {
  const chartText = document.querySelector(".chart-text2");
  const chart = document.querySelector(".chart2");
  const percen = document.querySelector(".chart-text2");

  let inner_text = percen.innerText;
  const regex = /\d+(\.\d+)?/;
  // Simulate data update (replace with your data source)
  let percentage = parseFloat(inner_text.match(regex)[0]);
  console.log(percentage);

  chart.style.width = `${(percentage / 100) * 200}px`;
  chart.style.height = `${(percentage / 100) * 200}px`;

  chartText.textContent = percentage + "%";
}
chart2();


function handleFile(files) {
  const file = files[0];
  const reader = new FileReader();

  reader.onload = function(event) {
      const contents = event.target.result;
      if (file.name.endsWith('.txt')) {
          document.getElementById('textarea').value = contents;
      } else if (file.name.endsWith('.docx')) {
          mammoth.extractRawText({arrayBuffer: contents}).then(function(result) {
              document.getElementById('textarea').value = result.value;
          }).catch(function(err) {
              console.log(err);
              alert("Error reading .docx file.");
          });
      } else if (file.name.endsWith('.pdf')) {
          readPdf(contents);
      }
  };

  reader.onerror = function(event) {
      console.error("File could not be read! Code " + event.target.error.code);
  };

  if (file) {
      if (file.name.endsWith('.docx') || file.name.endsWith('.pdf')) {
          reader.readAsArrayBuffer(file);
      } else {
          reader.readAsText(file);
      }
  }
}

function readPdf(data) {
  const loadingTask = pdfjsLib.getDocument({data});
  loadingTask.promise.then(function(pdf) {
      const pageNum = 1; // Change to read from multiple pages if needed
      pdf.getPage(pageNum).then(function(page) {
          page.getTextContent().then(function(textContent) {
              let text = '';
              for (const item of textContent.items) {
                  text += item.str + '\n';
              }
              document.getElementById('textarea').value = text;
          });
      });
  }).catch(function(err) {
      console.error('Error loading PDF:', err);
      alert("Error reading .pdf file.");
  });
}

var acc = document.getElementsByClassName("accordion");
var i;

for (i = 0; i < acc.length; i++) {
  acc[i].addEventListener("click", function() {
    /* Toggle between adding and removing the "active" class,
    to highlight the button that controls the panel */
    this.classList.toggle("active");

    /* Toggle between hiding and showing the active panel */
    var panel = this.nextElementSibling;
    if (panel.style.display === "block") {
      panel.style.display = "none";
    } else {
      panel.style.display = "block";
    }
  });
}
