// Global variables to store state
let currentData = null;
let currentModifiedData = null;
let isViewingModified = true;
let currentYear = null;
let currentMonth = null;
let availableData = [];

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Load available data years and months
    loadAvailableData();
    
    // Set up event listeners
    setupEventListeners();
});

// Load available data years and months from the server
function loadAvailableData() {
    fetch('/available_data')
        .then(response => response.json())
        .then(data => {
            availableData = data;
            populateYearMonthDropdowns(data);
            
            // Load default data (first available dataset)
            if (data.length > 0) {
                currentYear = data[0].year;
                currentMonth = data[0].month;
                document.getElementById('year-select').value = currentYear;
                document.getElementById('month-select').value = currentMonth;
                loadPM25Data(currentYear, currentMonth);
            }
        })
        .catch(error => {
            showMessage('Error loading available data: ' + error.message, 'danger');
        });
}

// Populate year and month dropdown selectors
function populateYearMonthDropdowns(data) {
    // Get unique years
    const years = [...new Set(data.map(item => item.year))].sort();
    
    // Populate year dropdown
    const yearSelect = document.getElementById('year-select');
    yearSelect.innerHTML = '';
    years.forEach(year => {
        const option = document.createElement('option');
        option.value = year;
        option.textContent = year;
        yearSelect.appendChild(option);
    });
    
    // Set up event listener for year change
    yearSelect.addEventListener('change', function() {
        currentYear = parseInt(this.value);
        updateMonthDropdown(currentYear);
        if (document.getElementById('month-select').options.length > 0) {
            currentMonth = parseInt(document.getElementById('month-select').value);
            loadPM25Data(currentYear, currentMonth);
        }
    });
    
    // Initialize month dropdown with first year's months
    if (years.length > 0) {
        updateMonthDropdown(years[0]);
    }
}

// Update month dropdown based on selected year
function updateMonthDropdown(year) {
    const months = availableData
        .filter(item => item.year === parseInt(year))
        .map(item => item.month)
        .sort((a, b) => a - b);
    
    const monthSelect = document.getElementById('month-select');
    monthSelect.innerHTML = '';
    months.forEach(month => {
        const option = document.createElement('option');
        option.value = month;
        option.textContent = getMonthName(month);
        monthSelect.appendChild(option);
    });
}

// Get month name from month number (1-12)
function getMonthName(month) {
    const monthNames = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ];
    return monthNames[month - 1];
}

// Load PM2.5 data for the specified year and month
function loadPM25Data(year, month, viewModified = isViewingModified) {
    showMessage('Loading data...', 'info');
    
    fetch(`/get_pm25_data?year=${year}&month=${month}&use_modified=${viewModified}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            currentYear = year;
            currentMonth = month;
            isViewingModified = data.is_modified;
            
            // Store the original data for comparison
            if (!data.is_modified) {
                currentData = data.data;
                currentModifiedData = JSON.parse(JSON.stringify(data.data)); // Deep copy
            } else {
                currentModifiedData = data.data;
                // Load original data in the background for comparison
                fetch(`/get_pm25_data?year=${year}&month=${month}&use_modified=false`)
                    .then(response => response.json())
                    .then(origData => {
                        if (!origData.error) {
                            currentData = origData.data;
                        }
                    });
            }
            
            // Plot the data
            plotHeatmap(data.data, data.range, year, month, data.is_modified);
            
            showMessage(`Loaded PM2.5 data for ${getMonthName(month)} ${year}${data.is_modified ? ' (Modified)' : ''}`, 'success');
        })
        .catch(error => {
            showMessage('Error loading data: ' + error.message, 'danger');
        });
}

// Plot heatmap using Plotly
function plotHeatmap(data, dataRange, year, month, isModified) {
    const heatmapDiv = document.getElementById('heatmap');
    
    // Create the heatmap
    const heatmapData = [{
        z: data,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: true,
        zmin: dataRange.min,
        zmax: dataRange.max,
        colorbar: {
            title: 'PM2.5 (µg/m³)',
            titleside: 'right'
        }
    }];
    
    const layout = {
        title: `PM2.5 (µg/m³) - ${getMonthName(month)} ${year}${isModified ? ' (Modified)' : ' (Original)'}`,
        xaxis: {
            title: 'X Coordinate',
            scaleanchor: 'y',
            constrain: 'domain'
        },
        yaxis: {
            title: 'Y Coordinate',
            autorange: 'reversed' // To match the usual image orientation
        },
        margin: {
            l: 50,
            r: 50,
            b: 50,
            t: 50,
            pad: 4
        }
    };
    
    Plotly.newPlot(heatmapDiv, heatmapData, layout);
    
    // Add selection box when data is plotted
    updateSelectionBox();
}

// Set up all event listeners
function setupEventListeners() {
    // Month change event
    document.getElementById('month-select').addEventListener('change', function() {
        currentMonth = parseInt(this.value);
        loadPM25Data(currentYear, currentMonth);
    });
    
    // Slider value updates
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', function() {
            // Update the displayed value
            document.getElementById(this.id + '-value').textContent = this.value;
            
            // Update the selection box if it's a coordinate slider
            if (['x0-slider', 'x1-slider', 'y0-slider', 'y1-slider'].includes(this.id)) {
                updateSelectionBox();
            }
        });
    });
    
    // Apply modification button
    document.getElementById('apply-button').addEventListener('click', applyModification);
    
    // Save changes button
    document.getElementById('save-button').addEventListener('click', saveChanges);
    
    // Reset button
    document.getElementById('reset-button').addEventListener('click', resetToOriginal);
    
    // Toggle between original and modified view
    document.getElementById('toggle-button').addEventListener('click', toggleView);
}

// Update the selection box on the heatmap
function updateSelectionBox() {
    const x0 = parseInt(document.getElementById('x0-slider').value);
    const x1 = parseInt(document.getElementById('x1-slider').value);
    const y0 = parseInt(document.getElementById('y0-slider').value);
    const y1 = parseInt(document.getElementById('y1-slider').value);
    
    // Make sure x1 > x0 and y1 > y0
    if (x0 >= x1) {
        document.getElementById('x1-slider').value = x0 + 1;
        document.getElementById('x1-value').textContent = x0 + 1;
    }
    if (y0 >= y1) {
        document.getElementById('y1-slider').value = y0 + 1;
        document.getElementById('y1-value').textContent = y0 + 1;
    }
    
    // Add rectangle shape to highlight the selection
    const heatmapDiv = document.getElementById('heatmap');
    const shapes = [{
        type: 'rect',
        x0: x0 - 0.5,
        x1: x1 - 0.5,
        y0: y0 - 0.5,
        y1: y1 - 0.5,
        line: {
            color: 'rgba(255, 0, 0, 1)',
            width: 2
        },
        fillcolor: 'rgba(255, 0, 0, 0.1)'
    }];
    
    Plotly.relayout(heatmapDiv, {shapes: shapes});
}

// Apply modification to the selected region
function applyModification() {
    if (!currentModifiedData) {
        showMessage('No data loaded to modify', 'warning');
        return;
    }
    
    const x0 = parseInt(document.getElementById('x0-slider').value);
    const x1 = parseInt(document.getElementById('x1-slider').value);
    const y0 = parseInt(document.getElementById('y0-slider').value);
    const y1 = parseInt(document.getElementById('y1-slider').value);
    const factor = parseFloat(document.getElementById('factor-slider').value);
    
    // Make a copy to work with
    let dataToModify = JSON.parse(JSON.stringify(currentModifiedData));
    
    // Apply modification factor to the selected region
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
            if (y < dataToModify.length && x < dataToModify[y].length) {
                dataToModify[y][x] *= factor;
            }
        }
    }
    
    // Update the current modified data
    currentModifiedData = dataToModify;
    
    // Re-plot with modified data
    const range = {
        min: Math.min(...dataToModify.flat().filter(val => !isNaN(val))),
        max: Math.max(...dataToModify.flat().filter(val => !isNaN(val)))
    };
    
    plotHeatmap(dataToModify, range, currentYear, currentMonth, true);
    isViewingModified = true;
    
    showMessage(`Applied factor ${factor} to selected region`, 'success');
}

// Save the modified data to the server
function saveChanges() {
    if (!currentModifiedData) {
        showMessage('No data to save', 'warning');
        return;
    }
    
    showMessage('Saving changes...', 'info');
    
    fetch('/save_modified_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            year: currentYear,
            month: currentMonth,
            data: currentModifiedData
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showMessage(data.message, 'success');
        } else {
            showMessage('Error: ' + data.message, 'danger');
        }
    })
    .catch(error => {
        showMessage('Error saving data: ' + error.message, 'danger');
    });
}

// Reset to original data
function resetToOriginal() {
    if (!currentData) {
        showMessage('No original data to reset to', 'warning');
        return;
    }
    
    showMessage('Resetting to original data...', 'info');
    
    fetch('/reset_to_original', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            year: currentYear,
            month: currentMonth
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Reset to original data in memory
            currentModifiedData = JSON.parse(JSON.stringify(currentData));
            
            // Re-plot with original data
            const range = {
                min: Math.min(...currentData.flat().filter(val => !isNaN(val))),
                max: Math.max(...currentData.flat().filter(val => !isNaN(val)))
            };
            
            plotHeatmap(currentData, range, currentYear, currentMonth, false);
            isViewingModified = false;
            
            showMessage(data.message, 'success');
        } else {
            showMessage('Error: ' + data.message, 'danger');
        }
    })
    .catch(error => {
        showMessage('Error resetting data: ' + error.message, 'danger');
    });
}

// Toggle between original and modified view
function toggleView() {
    if (!currentData || !currentModifiedData) {
        showMessage('No data to toggle view', 'warning');
        return;
    }
    
    isViewingModified = !isViewingModified;
    
    // Re-plot with selected data
    const dataToShow = isViewingModified ? currentModifiedData : currentData;
    const range = {
        min: Math.min(...dataToShow.flat().filter(val => !isNaN(val))),
        max: Math.max(...dataToShow.flat().filter(val => !isNaN(val)))
    };
    
    plotHeatmap(dataToShow, range, currentYear, currentMonth, isViewingModified);
    
    showMessage(`Now viewing ${isViewingModified ? 'modified' : 'original'} data`, 'info');
}

// Show status messages to the user
function showMessage(message, type) {
    const messageArea = document.getElementById('message-area');
    messageArea.innerHTML = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    // Auto-dismiss success messages after 3 seconds
    if (type === 'success' || type === 'info') {
        setTimeout(() => {
            const alert = messageArea.querySelector('.alert');
            if (alert) {
                alert.classList.remove('show');
                setTimeout(() => messageArea.innerHTML = '', 150);
            }
        }, 3000);
    }
}