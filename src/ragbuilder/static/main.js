function validateSourceData() {
    const sourceData = $('#sourceData').val();

    $.ajax({
        type: "POST",
        url: "/check_source_data",
        contentType: "application/json",
        data: JSON.stringify({ sourceData: sourceData }),
        success: function (response) {
            if (response.valid) {
                console.log('Source data is valid.');
            } else {
                alert('Invalid source data. Please check the URL or file/ directory path.');
            }
        },
        error: function (error) {
            console.error('Error validating source data:', error);
        }
    });
}

// Show modal to capture user inputs and send it to backend.
$(document).ready(function () {
    $('#newProject').click(function () {
        $('#newProjectModal').modal('show')
    });
    var dataExists = "";
    var existingSynthDataPath = "";
    
    // Check if source data path is valid
    $('#sourceData').on('blur', function() {
        validateSourceData();
    });

    // Disable Compressors if contextualCompression is unselected
    $('#contextualCompression').change(function() {
        if (!this.checked) {
            $('#longContextReorder, #crossEncoderReranker, #embeddingsRedundantFilter, #embeddingsClusteringFilter, #llmChainFilter').prop('checked', false).prop('disabled', true);
        } else {
            $('#longContextReorder, #crossEncoderReranker, #embeddingsRedundantFilter, #embeddingsClusteringFilter, #llmChainFilter').prop('disabled', false);
        }
    });

    $('#nextStep1').click(function () {
        const sourceData = $('#sourceData').val();
        $.ajax({
            type: "POST",
            url: "/check_test_data",
            contentType: "application/json",
            data: JSON.stringify({ sourceData: sourceData }),
            success: function (response) {
                if (response.exists) {
                    existingSynthDataPath = response.path;
                    dataExists=`<p><strong>Test data exists for the provided source dataset’s hash.</strong><br>Path: ${response.path}</p>`
                    $('#hashLookupResult').html(`${dataExists}`);
                    $('#foundExistingSynthData').show();
                    $('#useExistingSynthData').prop('checked', true);
                    $('#generateSynthetic').prop('checked', false);
                    // Make foundexisting checked, and uncheck generate synthetic data
                    $('#advancedSettings').collapse('hide');
                } else {
                    $('#hashLookupResult').html('');
                    $('#advancedSettings').collapse('show');
                }
            },
            error: function (error) {
                console.error(error);
            }
        });

        if ($('#includeNonTemplated').is(':checked')) {
            $('#step1').hide();
            $('#step2').show();
        } else {
            $('#step1').hide();
            $('#step3').show();

        }
    });

    
    $('#nextStep2').click(function () {
        $('#step2').hide();
        $('#step3').show();
    });

    
    $('#nextStep3').click(function () {
        var testDataHtml=null;
        var customSelections="";
        var selectedOption = $('input[name="syntheticDataOptions"]:checked').val();

        if (selectedOption === 'reuse') {
            testDataHtml = `<p><strong>Re-use synthetic data:</strong> ${$('#useExistingSynthData').is(':checked')? '✔' : '☐'}</p>${dataExists}`;
        } else if (selectedOption === 'generate') {
            testDataHtml = `<p><strong>Generate synthetic data:</strong> ${$('#generateSynthetic').is(':checked')? '✔' : '☐'}</p>`;
        } else {
            const testDataPath = $('#testDataPath').val();
            testDataHtml = `<p><strong>Test Data Path:</strong> ${testDataPath}</p>`;
        }

        if ($('#includeNonTemplated').is(':checked')) {
            // console.log(existingSynthDataPath);
            customSelections=`
                <p><strong>Chunking Strategy:</strong></p>
                <ul>
                    <li>Markdown: ${$('#markdown').is(':checked')? '✔' : '☐'}</li>
                    <li>HTML: ${$('#html').is(':checked')? '✔' : '☐'}</li>
                    <li>Semantic: ${$('#semantic').is(':checked')? '✔' : '☐'}</li>
                    <li>Recursive: ${$('#recursive').is(':checked')? '✔' : '☐'}</li>
                    <li>Character: ${$('#character').is(':checked')? '✔' : '☐'}</li>
                </ul>
                <p><strong>Chunk Size:</strong></p>
                <ul>
                    <li>Min: ${$('#chunkMin').val()}</li>
                    <li>Max: ${$('#chunkMax').val()}</li>
                </ul>
                <p><strong>Embedding Model:</strong></p>
                <ul>
                    <li>text-embedding-3-small: ${$('#embeddingSmall').is(':checked')? '✔' : '☐'}</li>
                    <li>text-embedding-3-large: ${$('#embeddingLarge').is(':checked')? '✔' : '☐'}</li>
                    <li>text-embedding-ada-002: ${$('#embeddingAda').is(':checked')? '✔' : '☐'}</li>
                </ul>
                <p><strong>Vector DB:</strong> ${$('input[name="vectorDB"]:checked').attr('id')}</p>
                <p><strong>Retriever:</strong></p>
                <ul>
                    <li>Vector DB - Similarity Search: ${$('#vectorSimilarity').is(':checked')? '✔' : '☐'}</li>
                    <li>Vector DB - MMR: ${$('#vectorMMR').is(':checked')? '✔' : '☐'}</li>
                    <li>BM25 Retriever: ${$('#bm25Retriever').is(':checked')? '✔' : '☐'}</li>
                    <li>Multi Query Retriever: ${$('#multiQuery').is(':checked')? '✔' : '☐'}</li>
                    <li>Parent Document Retriever - Full Documents: ${$('#parentDocFullDoc').is(':checked')? '✔' : '☐'}</li>
                    <li>Parent Document Retriever - Large Chunks: ${$('#parentDocLargeChunk').is(':checked')? '✔' : '☐'}</li>
                </ul>
                <p><strong>Top k:</strong></p>
                <ul>
                    <li>5: ${$('#topK5').is(':checked')? '✔' : '☐'}</li>
                    <li>10: ${$('#topK10').is(':checked')? '✔' : '☐'}</li>
                    <li>20: ${$('#topK20').is(':checked')? '✔' : '☐'}</li>
                </ul>
                <p><strong>Compression:</strong></p>
                <ul>
                    <li>Contextual Compression: ${$('#contextualCompression').is(':checked')? '✔' : '☐'}</li>
                    <ul>
                        <li>Long Context Reorder: ${$('#longContextReorder').is(':checked')? '✔' : '☐'}</li>
                        <li>Cross Encoder Re-ranker: ${$('#crossEncoderReranker').is(':checked')? '✔' : '☐'}</li>
                        <li>Embedding Redundant Filter: ${$('#embeddingsRedundantFilter').is(':checked')? '✔' : '☐'}</li>
                        <li>Embedding Clustering Filter: ${$('#embeddingsClusteringFilter').is(':checked')? '✔' : '☐'}</li>
                        <li>LLM Chain Filter: ${$('#llmChainFilter').is(':checked')? '✔' : '☐'}</li>
                    </ul>        
                </ul>
                <p><strong>LLM:</strong></p>
                <ul>
                    <li>GPT-3.5 Turbo: ${$('#gpt35').is(':checked')? '✔' : '☐'}</li>
                    <li>GPT-4o: ${$('#gpt4o').is(':checked')? '✔' : '☐'}</li>
                    <li>GPT-4 Turbo: ${$('#gpt4Turbo').is(':checked')? '✔' : '☐'}</li>
                </ul>
            `
        }

        // Fill the review section with selections from all steps
        const selections = `
            <p><strong>Description:</strong> ${$('#description').val()}</p>
            <p><strong>Source data:</strong> ${$('#sourceData').val()}</p>
            <p><strong>Use Pre-defined RAG Templates:</strong> ${$('#compareTemplates').is(':checked')? '✔' : '☐'}</p>
            <p><strong>Create Custom RAG Configurations:</strong> ${$('#includeNonTemplated').is(':checked')? '✔' : '☐'}</p>
            ${customSelections}
            ${testDataHtml}
        `;
        $('#step3').hide();
        $('#step4').show();
        $('#review').html(selections);
    });

    $('#previousStep2').click(function () {
        $('#step2').hide();
        $('#step1').show();
    });

    $('#previousStep3').click(function () {
        $('#step3').hide();
        if ($('#includeNonTemplated').is(':checked')) {
            $('#step2').show();
        } else {
            $('#step1').show();
        }
    });

    $('#previousStep4').click(function () {
        $('#step4').hide();
        $('#step3').show();
    });

    $('input[name="syntheticDataOptions"]').change(function () {
        if ($('#existingTestData').is(':checked')) {
            $('#existingTestDataPath').show();
            $('#generateSyntheticSection').hide();
        } else if ($('#useExistingSynthData').is(':checked')) {
            $('#existingTestDataPath').hide();
            $('#generateSyntheticSection').hide();
        } else {
            $('#existingTestDataPath').hide();
            $('#generateSyntheticSection').show();
            $('#advancedSettings').collapse('show');
        }

    });

    $('#confirmSelections').click(function () {
        const projectData = {
            description: $('#description').val(),
            sourceData: $('#sourceData').val(),
            compareTemplates: $('#compareTemplates').is(':checked'),
            includeNonTemplated: $('#includeNonTemplated').is(':checked'),
            chunkingStrategy: {
                MarkdownHeaderTextSplitter: $('#markdown').is(':checked'),
                HTMLHeaderTextSplitter: $('#html').is(':checked'),
                SemanticChunker: $('#semantic').is(':checked'),
                RecursiveCharacterTextSplitter: $('#recursive').is(':checked'),
                CharacterTextSplitter: $('#character').is(':checked')
            },
            chunkSize: {
                min: $('#chunkMin').val(),
                max: $('#chunkMax').val()
            },
            embeddingModel: {
                "text-embedding-3-small": $('#embeddingSmall').is(':checked'),
                "text-embedding-3-large": $('#embeddingLarge').is(':checked'),
                "text-embedding-ada-002": $('#embeddingAda').is(':checked')
            },
            vectorDB: $('input[name="vectorDB"]:checked').attr('id'),
            // {
            //     "chromaDB": $('#chromaDB').is(':checked'),
            //     "faissDB": $('#faissDB').is(':checked'),
            //     "singleStoreDB": $('#singleStoreDB').is(':checked'),
            //     "pineconeDB": $('#pineconeDB').is(':checked')
            // },
            retriever: {
                vectorSimilarity: $('#vectorSimilarity').is(':checked'),
                vectorMMR: $('#vectorMMR').is(':checked'),
                bm25Retriever: $('#bm25Retriever').is(':checked'),
                multiQuery: $('#multiQuery').is(':checked'),
                parentDocFullDoc: $('#parentDocFullDoc').is(':checked'),
                parentDocLargeChunk: $('#parentDocLargeChunk').is(':checked')
            },
            topK: {
                search_k_5: $('#topK5').is(':checked'),
                search_k_10: $('#topK10').is(':checked'),
                search_k_20: $('#topK20').is(':checked')
            },
            contextualCompression: $('#contextualCompression').is(':checked'),
            llm: {
                "gpt-3.5-turbo": $('#gpt35').is(':checked'),
                "gpt-4o": $('#gpt4o').is(':checked'),
                "gpt-4-turbo": $('#gpt4Turbo').is(':checked')
            },
            generateSyntheticData: $('#generateSynthetic').is(':checked'),
            optimization: $('input[name="optimization"]:checked').attr('id')
        };
        
        if ($('#contextualCompression').is(':checked')) {
                projectData.compressors = {
                    LongContextReorder: $('#longContextReorder').is(':checked'),
                    CrossEncoderReranker: $('#crossEncoderReranker').is(':checked'),
                    EmbeddingsRedundantFilter: $('#embeddingsRedundantFilter').is(':checked'),
                    EmbeddingsClusteringFilter: $('#embeddingsClusteringFilter').is(':checked'),
                    LLMChainFilter: $('#llmChainFilter').is(':checked')
                };
        }
        if ($('#existingTestData').is(':checked')) {
            projectData.testDataPath = $('#testDataPath').val();
        } else if ($('#useExistingSynthData').is(':checked')) {
            projectData.existingSynthDataPath=`${existingSynthDataPath}`;
        } else {
            projectData.syntheticDataGeneration = {
                testSize: $('#testSize').val(),
                criticLLM: $('#criticLLM').val(),
                generatorLLM: $('#generatorLLM').val(),
                embedding: $('#embedding').val()
            };
        }
    
        console.log(JSON.stringify(projectData));

        $('#step4').hide();
        $('#progressSection').show();

        fetchLogUpdates();

        $.ajax({
            type: "POST",
            url: "/rbuilder",
            contentType: "application/json",
            data: JSON.stringify(projectData),
            success: function(response) {
                if (response.status === "success") {
                    alert(response.message);
                    // Redirect to the summary page
                    window.location.href = "/summary/" + response.run_id;
                    // Refresh the page after a short delay
                    setTimeout(function() {
                        window.location.reload();
                    }, 1000); // Delay in milliseconds
                } else {
                    alert("Unexpected response: " + JSON.stringify(response));
                }
            },
            error: function(error) {
                var errorMsg = "An error occurred. Please try again.";
                if (error.responseJSON && error.responseJSON.message) {
                    errorMsg = error.responseJSON.message;
                }
                alert(errorMsg);
            }
        });            

        // Tooltip logic
        $('.info-icon').hover(function() {
            $(this).next('.tooltip').show();
        }, function() {
            $(this).next('.tooltip').hide();
        });
    });
    
    function fetchLogUpdates() {
        $.ajax({
            type: "GET",
            url: "/get_log_filename",
            success: function (response) {
                const logFilename = response.log_filename;
                const logInterval = setInterval(function () {
                    $.ajax({
                        type: "GET",
                        url: "/get_log_updates",
                        success: function (response) {
                            $('#logOutput').text(response.log_content);
                            const logContent = response.log_content;

                            // Example: If the log content includes "Processing finished successfully." stop the interval
                            if (logContent.includes("Processing finished successfully.")) {
                                clearInterval(logInterval);
                                $('#progressSection').hide();
                                $('#completionSection').show();
                            }

                            // Update progress bar (this is just a simple example, you can enhance it as needed)
                            const progressText = logContent.split("\n").length;
                            const progressPercentage = Math.min((progressText / 200) * 100, 100);
                            $('#progressBar').css('width', progressPercentage + '%').attr('aria-valuenow', progressPercentage);
                        },
                        error: function (error) {
                            console.error(error);
                        }
                    });
                }, 2000); // Update every 2 seconds
            },
            error: function (error) {
                console.error(error);
            }
        });
    }
});



