document.addEventListener('DOMContentLoaded', () => {
    // DOM References - UI Controls
    const checkButton = document.getElementById('checkButton');
    const newsTextArea = document.getElementById('newsText');
    const resultArea = document.getElementById('resultArea');
    const errorArea = document.getElementById('errorArea');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorMessageP = document.getElementById('errorMessage');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    
    // Tab Navigation
    const tabLinks = document.querySelectorAll('.sidebar nav ul li');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Local storage for history and statistics
    const historyKey = 'fakeNewsHistory';
    const statsKey = 'fakeNewsStats';
    let analysisHistory = JSON.parse(localStorage.getItem(historyKey)) || [];
    let statistics = JSON.parse(localStorage.getItem(statsKey)) || {
        totalAnalyses: 0,
        fakeNewsCount: 0,
        realNewsCount: 0,
        confidenceSum: 0,
        sentimentCounts: {
            positive: 0,
            neutral: 0,
            negative: 0
        }
    };
    
    // Initialize UI
    updateHistoryUI();
    updateStatisticsUI();
    
    // Tab switching functionality
    tabLinks.forEach(tabLink => {
        tabLink.addEventListener('click', () => {
            const tabId = tabLink.getAttribute('data-tab');
            
            // Update active tab link
            tabLinks.forEach(link => link.classList.remove('active'));
            tabLink.classList.add('active');
            
            // Show correct tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabId}-tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
    
    // Clear history button
    clearHistoryBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear all history items?')) {
            analysisHistory = [];
            localStorage.setItem(historyKey, JSON.stringify(analysisHistory));
            updateHistoryUI();
        }
    });

    // Main analysis functionality
    checkButton.addEventListener('click', async () => {
        const text = newsTextArea.value.trim();

        // Get references to error and indicator elements
        const errorArea = document.getElementById('errorArea');
        const errorMessageP = document.getElementById('errorMessage');
        const initialLoadingIndicator = document.getElementById('loadingIndicator');
        const webScrapingIndicator = document.getElementById('webScrapingIndicator');
        const llmLoadingIndicator = document.getElementById('llmLoadingIndicator');
        const resultArea = document.getElementById('resultArea');
        const llmResultSection = document.getElementById('llmResult');

        // Clear previous results/errors and hide all sections
        resultArea.style.display = 'none';
        errorArea.style.display = 'none';
        webScrapingIndicator.style.display = 'none';
        llmLoadingIndicator.style.display = 'none';
        initialLoadingIndicator.style.display = 'flex';
        errorMessageP.textContent = '';
        
        if (llmResultSection) {
            llmResultSection.style.display = 'none';
        }

        // Clear all result elements
        document.getElementById('analyzedText').textContent = '';
        document.getElementById('prediction').textContent = '';
        document.getElementById('probability').textContent = '0';
        document.querySelector('.prediction-badge').removeAttribute('data-result');
        document.getElementById('scrapingStatus').textContent = '';
        document.getElementById('scrapedLinksList').innerHTML = '';
        document.getElementById('sentimentLabel').textContent = 'N/A';
        document.getElementById('sentimentScore').textContent = 'N/A';
        document.getElementById('sentimentMeter').style.width = '50%';
        document.getElementById('linguisticFlagsList').innerHTML = '';
        document.getElementById('similarTrueList').innerHTML = '';
        document.getElementById('similarFakeList').innerHTML = '';
        document.getElementById('scrapedKeywords').innerHTML = '';
        document.getElementById('scrapedAnalysisSourcesList').innerHTML = '';
        document.getElementById('llmAnalysisText').textContent = '';
        
        // Hide AI loader initially
        const aiLoader = document.querySelector('.ai-loader');
        if (aiLoader) aiLoader.style.display = 'none';

        if (!text) {
            errorMessageP.textContent = 'Please paste some news text into the text area.';
            errorArea.style.display = 'block';
            initialLoadingIndicator.style.display = 'none';
            return;
        }

        checkButton.disabled = true;
        checkButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

        try {
            // PHASE 1: Initial loading (1 second)
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // PHASE 2: Switch to web scraping indicator
            initialLoadingIndicator.style.display = 'none';
            webScrapingIndicator.style.display = 'flex';
            
            // Wait a moment to show the web scraping indicator
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Make API call to the backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();

            // PHASE 3: Display web scraping results
            webScrapingIndicator.style.display = 'none';
            displayWebScrapingResults(result);
            
            // Show results container
            resultArea.style.display = 'block';
            
            // Force browser to render the updates
            void resultArea.offsetHeight;
            
            // PHASE 4: Start LLM analysis with indicator
            llmLoadingIndicator.style.display = 'flex';
            
            // Let the user see the web results for a moment
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // PHASE 5: Display LLM results
            llmLoadingIndicator.style.display = 'none';
            displayLLMResults(result.llm_analysis || "LLM analysis was not returned or failed.", false);
            
            // Save to history
            saveToHistory(text.substring(0, 200), result);
            
            // Update statistics
            updateStatistics(result);

        } catch (error) {
            console.error('Error during analysis fetch or processing:', error);
            errorMessageP.textContent = `An error occurred: ${error.message}`;
            errorArea.style.display = 'block';
            
            // Hide all indicators
            initialLoadingIndicator.style.display = 'none';
            webScrapingIndicator.style.display = 'none';
            llmLoadingIndicator.style.display = 'none';
        } finally {
            checkButton.disabled = false;
            checkButton.innerHTML = '<i class="fas fa-magnifying-glass"></i> Analyze News';
        }
    });

    // Helper function to populate a simple list (ul element) with text items
    function populateSimpleList(listElement, items) {
        listElement.innerHTML = ''; // Clear existing items
        if (items && items.length > 0) {
            items.forEach(itemText => {
                const li = document.createElement('li');
                // Basic check to prevent potential HTML injection
                li.textContent = typeof itemText === 'string' ? itemText : JSON.stringify(itemText);
                listElement.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'None found or analysis not applicable.';
            li.style.fontStyle = 'italic';
            li.style.color = '#888';
            listElement.appendChild(li);
        }
    }

    // Helper function to populate scraped links with improved styling
    function populateLinksList(listElement, scrapedResults) {
        listElement.innerHTML = ''; // Clear first
        if (scrapedResults.status === 'success' && scrapedResults.data && scrapedResults.data.length > 0) {
            scrapedResults.data.forEach(item => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = item.link;
                a.textContent = item.title;
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
                li.appendChild(a);
                const span = document.createElement('span');
                span.textContent = item.snippet;
                li.appendChild(span);
                listElement.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = scrapedResults.message || 'No related articles found or search failed.';
            if (scrapedResults.status === 'error' || scrapedResults.status === 'skipped') {
                li.style.fontStyle = 'italic';
                li.style.color = '#888';
            }
            listElement.appendChild(li);
        }
    }

    // Function to render keywords as interactive tags
    function renderKeywordTags(container, keywords) {
        container.innerHTML = '';
        
        if (keywords && keywords.length > 0) {
            keywords.forEach(keyword => {
                const tag = document.createElement('span');
                tag.className = 'keyword-tag';
                tag.textContent = keyword;
                container.appendChild(tag);
            });
        } else {
            container.textContent = 'No keywords identified or analysis failed.';
        }
    }

    // Function to create typing animation effect for LLM analysis
    function typingAnimation(element, text, speed = 30) {
        // Clear any existing text
        element.textContent = '';
        const cursor = element.parentElement.querySelector('.cursor');
        const container = element.parentElement;
        const aiLoader = container.querySelector('.ai-loader');
        
        // Show the AI loader while "thinking"
        if (aiLoader) aiLoader.style.display = 'flex';
        
        // Hide the cursor initially
        if (cursor) cursor.style.display = 'none';
        
        // Make sure the text is valid and not cut off
        if (!text || typeof text !== 'string') {
            text = "Analysis result unavailable or incomplete.";
        }
        
        // Ensure the text ends with proper punctuation to avoid looking cut off
        if (text.length > 0 && !text.endsWith('.') && !text.endsWith('!') && !text.endsWith('?')) {
            text += '.';
        }
        
        // Think for a moment before typing (simulate AI thinking)
        setTimeout(() => {
            // Hide loader when typing starts
            if (aiLoader) aiLoader.style.display = 'none';
            
            // Show cursor
            if (cursor) cursor.style.display = 'inline-block';
            
            let i = 0;
            let lastPauseIndex = 0;
            
            const typeChar = () => {
                if (i < text.length) {
                    // Add character
                    element.textContent += text.charAt(i);
                    i++;
                    
                    // Natural typing speed variations
                    const isEndOfSentence = text.charAt(i-1) === '.' || text.charAt(i-1) === '?' || text.charAt(i-1) === '!';
                    const isComma = text.charAt(i-1) === ',';
                    const charactersTyped = i - lastPauseIndex;
                    
                    let nextDelay = speed;
                    
                    // Add natural pauses
                    if (isEndOfSentence) {
                        nextDelay = speed * 5; // Longer pause after sentences
                        lastPauseIndex = i;
                    } else if (isComma) {
                        nextDelay = speed * 3; // Medium pause after commas
                    } else if (charactersTyped > 50 && Math.random() > 0.8) {
                        nextDelay = speed * 2; // Random small pause while typing
                        lastPauseIndex = i;
                    }
                    
                    // Scroll the container to keep up with the text
                    container.scrollTop = container.scrollHeight;
                    
                    setTimeout(typeChar, nextDelay);
                } else {
                    // Hide cursor when done typing
                    if (cursor) cursor.style.display = 'none';
                    
                    // Add a complete indicator to show the analysis is fully displayed
                    const completeIndicator = document.createElement('div');
                    completeIndicator.className = 'analysis-complete';
                    completeIndicator.innerHTML = '<i class="fas fa-check-circle"></i> Analysis complete';
                    container.appendChild(completeIndicator);
                }
            };
            
            // Start typing
            setTimeout(typeChar, 300);
        }, 1500); // Thinking animation duration
    }

    // Function to save analysis to history
    function saveToHistory(snippet, result) {
        // Make sure we store a clean, complete copy of the result
        const resultCopy = JSON.parse(JSON.stringify(result));
        
        const historyItem = {
            id: Date.now(), // Unique ID based on timestamp
            date: new Date().toISOString(),
            snippet: snippet,
            classification: result.classification || 'Unknown',
            confidence: result.classification_confidence 
                ? (result.classification_confidence * 100).toFixed(2) 
                : 0,
            sentiment: result.sentiment.label || 'Unknown',
            fullText: newsTextArea.value.trim(), // Save full text for reloading
            fullResult: resultCopy // Store complete result for reloading
        };
        
        // Add to beginning of array (newest first)
        analysisHistory.unshift(historyItem);
        
        // Limit history to 50 items to prevent localStorage overflow
        if (analysisHistory.length > 50) {
            analysisHistory = analysisHistory.slice(0, 50);
        }
        
        // Save to localStorage
        localStorage.setItem(historyKey, JSON.stringify(analysisHistory));
        
        // Update UI if we're on the history tab
        updateHistoryUI();
    }

    // Function to update statistics based on result
    function updateStatistics(result) {
        // Update counts
        statistics.totalAnalyses++;
        
        // Fix case sensitivity issue - compare case-insensitive
        const classification = result.classification ? result.classification.toUpperCase() : '';
        if (classification === 'FAKE') {
            statistics.fakeNewsCount++;
        } else if (classification === 'REAL') {
            statistics.realNewsCount++;
        }
        
        // Add confidence to running sum for average calculation
        if (result.classification_confidence) {
            statistics.confidenceSum += result.classification_confidence * 100;
        }
        
        // Update sentiment counts
        if (result.sentiment && result.sentiment.label) {
            const sentiment = result.sentiment.label.toLowerCase();
            if (sentiment === 'positive') {
                statistics.sentimentCounts.positive++;
            } else if (sentiment === 'negative') {
                statistics.sentimentCounts.negative++;
            } else {
                statistics.sentimentCounts.neutral++;
            }
        }
        
        // Save to localStorage
        localStorage.setItem(statsKey, JSON.stringify(statistics));
        
        // Update stats UI
        updateStatisticsUI();
    }

    // Function to update history UI
    function updateHistoryUI() {
        const historyList = document.getElementById('historyList');
        
        if (!historyList) return;
        
        // Clear current list
        historyList.innerHTML = '';
        
        if (analysisHistory.length === 0) {
            // Show empty state
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-state';
            emptyState.innerHTML = `
                <i class="fas fa-history empty-icon"></i>
                <p>No history items yet. Start analyzing news to build your history.</p>
            `;
            historyList.appendChild(emptyState);
            return;
        }
        
        // Create history items
        analysisHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.dataset.id = item.id;
            
            // Format date for display
            const date = new Date(item.date);
            // Format like: 5/6/2023 3:08:20 PM
            const formattedDate = `${date.getMonth() + 1}/${date.getDate()}/${date.getFullYear()} ${formatTime(date)}`;
            
            // Create prediction class based on result
            const predictionClass = item.classification.toLowerCase() === 'fake' ? 'fake' : 
                                    item.classification.toLowerCase() === 'real' ? 'real' : '';
            
            // Create a short title from the snippet
            const title = item.snippet.split(' ').slice(0, 5).join(' ');
            
            historyItem.innerHTML = `
                <h3>${title}</h3>
                <div class="history-meta">
                    <span>${formattedDate}</span>
                    <div>
                        <span class="history-prediction ${predictionClass}">
                            ${item.classification} (based on dataset)
                        </span>
                        <div>Confidence: ${item.confidence}%</div>
                    </div>
                </div>
                <div class="history-actions">
                    <button class="reload-btn"><i class="fas fa-redo-alt"></i> Reload</button>
                    <button class="delete-btn"><i class="fas fa-trash"></i></button>
                </div>
            `;
            
            // Add to the list
            historyList.appendChild(historyItem);
            
            // Add event listeners to buttons
            const reloadBtn = historyItem.querySelector('.reload-btn');
            const deleteBtn = historyItem.querySelector('.delete-btn');
            
            // Reload button loads the analysis back into the detector tab
            reloadBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent the parent click from firing
                reloadHistoryItem(item);
            });
            
            // Delete button removes just this item
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent the parent click from firing
                deleteHistoryItem(item.id);
            });
            
            // Click on the item itself also reloads it
            historyItem.addEventListener('click', () => {
                reloadHistoryItem(item);
            });
        });
    }

    // Helper function to format time with AM/PM
    function formatTime(date) {
        let hours = date.getHours();
        const minutes = date.getMinutes();
        const seconds = date.getSeconds();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        
        hours = hours % 12;
        hours = hours ? hours : 12; // the hour '0' should be '12'
        
        const formattedMinutes = minutes < 10 ? '0' + minutes : minutes;
        const formattedSeconds = seconds < 10 ? '0' + seconds : seconds;
        
        return `${hours}:${formattedMinutes}:${formattedSeconds} ${ampm}`;
    }

    // Function to reload a history item
    function reloadHistoryItem(item) {
        // Switch to detector tab
        tabLinks.forEach(link => {
            if (link.getAttribute('data-tab') === 'detector') {
                link.click();
            }
        });
        
        // Set the textarea value to the stored text
        if (item.fullText) {
            newsTextArea.value = item.fullText;
            
            // Display the results without making a new API call
            if (item.fullResult) {
                // Use the history display function instead
                displayHistoryResults(item.fullResult);
            } else {
                // If we don't have the full result, trigger a new analysis
                checkButton.click();
            }
        } else {
            // Legacy fallback if we don't have the full text
            newsTextArea.value = item.snippet;
            checkButton.click();
        }
    }

    // Function to delete a history item
    function deleteHistoryItem(id) {
        if (confirm('Are you sure you want to delete this history item?')) {
            // Filter out the item with the given id
            analysisHistory = analysisHistory.filter(item => item.id !== id);
            
            // Save to localStorage
            localStorage.setItem(historyKey, JSON.stringify(analysisHistory));
            
            // Update the UI
            updateHistoryUI();
        }
    }

    // Function to display web scraping results
    function displayWebScrapingResults(result) {
        // Get references to dynamic result elements for non-LLM results
        const analyzedTextSpan = document.getElementById('analyzedText');
        const predictionSpan = document.getElementById('prediction');
        const probabilitySpan = document.getElementById('probability');
        const predictionBadge = document.querySelector('.prediction-badge');
        const scrapingStatusSpan = document.getElementById('scrapingStatus');
        const scrapedLinksList = document.getElementById('scrapedLinksList');
        const sentimentLabelSpan = document.getElementById('sentimentLabel');
        const sentimentScoreSpan = document.getElementById('sentimentScore');
        const sentimentMeter = document.getElementById('sentimentMeter');
        const linguisticFlagsList = document.getElementById('linguisticFlagsList');
        const similarTrueList = document.getElementById('similarTrueList');
        const similarFakeList = document.getElementById('similarFakeList');
        const scrapedKeywordsDiv = document.getElementById('scrapedKeywords');
        const scrapedAnalysisSourcesList = document.getElementById('scrapedAnalysisSourcesList');
        
        // Hide the LLM section completely
        const llmResultSection = document.getElementById('llmResult');
        if (llmResultSection) {
            llmResultSection.style.display = 'none';
        }
        
        // Set data attribute for styling based on prediction
        const prediction = result.classification || 'UNKNOWN';
        predictionBadge.setAttribute('data-result', prediction);
        
        // Update sentiment meter position (0-1 range)
        const sentimentScore = result.sentiment.score !== undefined ? result.sentiment.score : 0;
        // Convert -1 to 1 scale to 0 to 100% scale for the meter
        const meterPosition = ((sentimentScore + 1) / 2) * 100;
        sentimentMeter.style.width = `${meterPosition}%`;

        // Populate results
        analyzedTextSpan.textContent = result.analyzed_text.substring(0, 150);
        predictionSpan.textContent = prediction;
        probabilitySpan.textContent = result.classification_confidence
            ? (result.classification_confidence * 100).toFixed(2)
            : 'N/A';
        sentimentLabelSpan.textContent = result.sentiment.label || 'Error';
        sentimentScoreSpan.textContent = result.sentiment.score !== undefined ? result.sentiment.score.toFixed(3) : 'N/A';
        populateSimpleList(linguisticFlagsList, result.basic_linguistic_flags || ["Analysis Error."]);
        populateSimpleList(similarTrueList, result.similarity_results.similar_true || ["No similar true articles found or error."]);
        populateSimpleList(similarFakeList, result.similarity_results.similar_fake || ["No similar fake articles found or error."]);
        scrapingStatusSpan.textContent = result.scraped_results.message || 'Status unknown';
        
        // Populate search results with improved styling
        populateLinksList(scrapedLinksList, result.scraped_results);

        // Render keywords as tags in keyword cloud
        renderKeywordTags(scrapedKeywordsDiv, result.scraped_analysis?.keywords);
        
        // Populate sources list
        scrapedAnalysisSourcesList.innerHTML = '';
        if (result.scraped_analysis && result.scraped_analysis.sources && result.scraped_analysis.sources.length > 0) {
            result.scraped_analysis.sources.forEach(item => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = item.link;
                a.textContent = item.title;
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
                li.appendChild(a);
                scrapedAnalysisSourcesList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No sources listed for analysis.';
            scrapedAnalysisSourcesList.appendChild(li);
        }
    }

    // Function for displaying history items without animations
    function displayHistoryResults(result) {
        // Display all results at once without animations
        displayWebScrapingResults(result);
        
        // Show the result container
        document.getElementById('resultArea').style.display = 'block';
        
        // Show LLM text without animation
        const llmResultSection = document.getElementById('llmResult');
        const llmAnalysisTextP = document.getElementById('llmAnalysisText');
        const aiLoader = document.querySelector('.ai-loader');
        
        if (llmResultSection) {
            llmResultSection.style.display = 'block';
        }
        
        if (llmAnalysisTextP) {
            llmAnalysisTextP.textContent = result.llm_analysis || "LLM analysis was not returned or failed.";
        }
        
        if (aiLoader) {
            aiLoader.style.display = 'none';
        }
    }

    // Function to display LLM results with animation
    function displayLLMResults(llmText, skipAnimation) {
        const llmResultSection = document.getElementById('llmResult');
        const llmAnalysisTextP = document.getElementById('llmAnalysisText');
        const aiLoader = document.querySelector('.ai-loader');
        
        // Make the LLM section visible with an animation
        if (llmResultSection) {
            // Add animation classes
            llmResultSection.style.opacity = '0';
            llmResultSection.style.transform = 'translateY(20px)';
            llmResultSection.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            llmResultSection.style.display = 'block';
            
            // Force reflow
            void llmResultSection.offsetHeight;
            
            // Apply animation
            llmResultSection.style.opacity = '1';
            llmResultSection.style.transform = 'translateY(0)';
        }
        
        if (skipAnimation) {
            // Just display the text immediately
            llmAnalysisTextP.textContent = llmText;
            if (aiLoader) aiLoader.style.display = 'none';
        } else {
            // Show the AI loader
            if (aiLoader) aiLoader.style.display = 'flex';
            
            // Start typing animation
            setTimeout(() => {
                typingAnimation(llmAnalysisTextP, llmText);
            }, 800);
        }
    }

    // Function to update statistics UI and charts
    function updateStatisticsUI() {
        // Update stat counters
        document.getElementById('totalAnalyses').textContent = statistics.totalAnalyses;
        document.getElementById('fakeNewsCount').textContent = statistics.fakeNewsCount;
        document.getElementById('realNewsCount').textContent = statistics.realNewsCount;
        
        // Calculate and update average confidence
        const avgConfidence = statistics.totalAnalyses > 0
            ? (statistics.confidenceSum / statistics.totalAnalyses).toFixed(2)
            : '0';
        document.getElementById('avgConfidence').textContent = avgConfidence + '%';
        
        // Create/update charts if we have data and Chart.js is loaded
        if (statistics.totalAnalyses > 0 && typeof Chart !== 'undefined') {
            createResultsDistributionChart();
            createSentimentDistributionChart();
        }
    }

    // Create the results distribution chart (Fake vs Real)
    function createResultsDistributionChart() {
        const ctx = document.getElementById('resultsDistributionChart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (window.resultsChart instanceof Chart) {
            window.resultsChart.destroy();
        }
        
        // Create new chart
        window.resultsChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Fake News', 'Real News', 'Undetermined'],
                datasets: [{
                    data: [
                        statistics.fakeNewsCount,
                        statistics.realNewsCount,
                        statistics.totalAnalyses - statistics.fakeNewsCount - statistics.realNewsCount
                    ],
                    backgroundColor: [
                        '#ef4444', // Red for fake
                        '#10b981', // Green for real
                        '#6b7280'  // Gray for undetermined
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    // Create the sentiment distribution chart
    function createSentimentDistributionChart() {
        const ctx = document.getElementById('sentimentDistributionChart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (window.sentimentChart instanceof Chart) {
            window.sentimentChart.destroy();
        }
        
        // Create new chart
        window.sentimentChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    label: 'Sentiment Distribution',
                    data: [
                        statistics.sentimentCounts.positive,
                        statistics.sentimentCounts.neutral,
                        statistics.sentimentCounts.negative
                    ],
                    backgroundColor: [
                        '#10b981', // Green for positive
                        '#f59e0b', // Yellow for neutral
                        '#ef4444'  // Red for negative
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
}); 