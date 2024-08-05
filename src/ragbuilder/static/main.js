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

let progressInterval;
let smoothInterval;

$(document).ready(function () {
    // Show modal to capture user inputs and send it to backend.
    $('#newProject').click(function () {
        $('#newProjectModal').modal('show')
    });
    var dataExists = "";
    var existingSynthDataPath = "";
    var chunk_size_values = [500, 2000]
    
    // Check if source data path is valid
    $('#sourceData').on('blur', function() {
        validateSourceData();
    });

    document.querySelectorAll('[data-mdb-toggle="collapse"]').forEach(element => {
        new mdb.Collapse(element);
    });

    // Handle parent checkbox selection
    $('.treeview-category > input[type="checkbox"]').change(function () {
        var isChecked = $(this).prop('checked');
        $(this).closest('li').find('input[type="checkbox"]').prop('checked', isChecked);
    });

    // Handle child checkbox selection
    $('.treeview-category li input[type="checkbox"]').change(function () {
        var allSiblings = $(this).closest('ul').find('> li > input[type="checkbox"]');
        var allChecked = allSiblings.length === allSiblings.filter(':checked').length;
        var parentCheckbox = $(this).closest('ul').siblings('input[type="checkbox"]');
        parentCheckbox.prop('checked', allChecked);
    });

    // Handle chevron icon toggle and collapse/expand
    $('.treeview-category > .chevron-icon').click(function () {
        var $this = $(this);
        var target = $this.closest('a').data('mdb-target');
        $(target).collapse('toggle');
        $this.closest('a').attr('aria-expanded', function (index, attr) {
            return attr === 'true' ? 'false' : 'true';
        });
        $this.closest('a').toggleClass('collapsed');
    });

    // Initialize noUiSlider
    const slider = document.getElementById('chunk-size-multi-slider');
    noUiSlider.create(slider, {
        start: [500, 2000],
        connect: [false, true, false],
        range: {
            'min': 100,
            'max': 5000
        },
        step: 100,
        tooltips: true,
        format: {
            to: function (value) {
                return Math.round(value);
            },
            from: function (value) {
                return Number(value);
            }
        }
    });

    slider.noUiSlider.on('update', function (values, handle) {
        document.getElementById('minValue').innerText = values[0];
        document.getElementById('maxValue').innerText = values[1];
    });
    
    $('#embeddingHuggingFace').change(function () {
        if (this.checked) {
            $('#embeddingHuggingFaceModelDiv').show();
        } else {
            $('#embeddingHuggingFaceModelDiv').hide();
            $('#embeddingHuggingFaceModel').val('');
        }
    });

    $('#embeddingAzureOAI').change(function () {
        if (this.checked) {
            $('#embeddingAzureOAIModelDiv').show();
        } else {
            $('#embeddingAzureOAIModelDiv').hide();
            $('#embeddingAzureOAIModel').val('');
        }
    });

    $('#embeddingGoogleVertexAI').change(function () {
        if (this.checked) {
            $('#embeddingGoogleVertexAIModelDiv').show();
        } else {
            $('#embeddingGoogleVertexAIModelDiv').hide();
            $('#embeddingGoogleVertexAIModel').val('');
        }
    });

    $('#llmHuggingFace').change(function () {
        if (this.checked) {
            $('#llmHuggingFaceModelDiv').show();
        } else {
            $('#llmHuggingFaceModelDiv').hide();
            $('#llmHuggingFaceModel').val('');
        }
    });

    $('#llmGroq').change(function () {
        if (this.checked) {
            $('#llmGroqModelDiv').show();
        } else {
            $('#llmGroqModelDiv').hide();
            $('#llmGroqModel').val('');
        }
    });

    $('#llmAzureOAI').change(function () {
        if (this.checked) {
            $('#llmAzureOAIModelDiv').show();
        } else {
            $('#llmAzureOAIModelDiv').hide();
            $('#llmAzureOAIModel').val('');
        }
    });

    $('#llmGoogleVertexAI').change(function () {
        if (this.checked) {
            $('#llmGoogleVertexAIModelDiv').show();
        } else {
            $('#llmGoogleVertexAIModelDiv').hide();
            $('#llmGoogleVertexAIModel').val('');
        }
    });

    // Disable Compressors if contextualCompression is unselected
    $('#contextualCompression').change(function() {
        if (!this.checked) {
            $('#longContextReorder, #crossEncoderReranker, #embeddingsRedundantFilter, #embeddingsClusteringFilter, #llmChainFilter').prop('checked', false).prop('disabled', true);
        } else {
            $('#longContextReorder, #crossEncoderReranker, #embeddingsRedundantFilter, #embeddingsClusteringFilter, #llmChainFilter').prop('disabled', false);
        }
    });

    // Show or hide the number of runs input based on the selected optimization option
    $('input[name="optimization"]').change(function () {
        if ($('#bayesianOptimization').is(':checked')) {
            $('#numRunsContainer').show();
        } else {
            $('#numRunsContainer').hide();
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
            testDataHtml = `<p><strong>Re-use synthetic data:</strong> ${$('#useExistingSynthData').is(':checked')? '<i class="fas fa-check-circle me-2 text-success"></i>' : '<i class="fa-regular fa-circle me-2 text-secondary"></i>'}</p>${dataExists}`;
        } else if (selectedOption === 'generate') {
            testDataHtml = `<p><strong>Generate synthetic data:</strong> ${$('#generateSynthetic').is(':checked')? '<i class="fas fa-check-circle me-2 text-success"></i>' : '<i class="fa-regular fa-circle me-2 text-secondary"></i>'}</p>`;
        } else {
            const testDataPath = $('#testDataPath').val();
            testDataHtml = `<p><strong>Test Data Path:</strong> ${testDataPath}</p>`;
        }

        if ($('#includeNonTemplated').is(':checked')) {
            // console.log(existingSynthDataPath);
            chunk_size_values = slider.noUiSlider.get();
            customSelections=`
                <div class="row row-cols-2">
                    <div class="col-md-6 mt-3">
                        <p><strong>Chunking Strategy:</strong></p>
                        <ul>
                            <li><div class="row"><div class="col-8">Markdown: </div>${$('#markdown').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">HTML: </div>${$('#html').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Semantic: </div>${$('#semantic').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Recursive: </div>${$('#recursive').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Character: </div>${$('#character').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                        </ul>
                    </div>
                    <div class="col-md-6 mt-3">
                        <p><strong>Chunk Size:</strong></p>
                        <ul>
                            <li>Min: ${chunk_size_values[0]}</li>
                            <li>Max: ${chunk_size_values[1]}</li>
                        </ul>
                    </div>
                    <div class="col-md-6 mt-3">
                        <p><strong>Embedding Model:</strong></p>
                        <ul>
                            <li><div class="row"><div class="col-8">text-embedding-3-small: </div>${$('#embeddingSmall').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div</li>
                            <li><div class="row"><div class="col-8">text-embedding-3-large: </div>${$('#embeddingLarge').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">text-embedding-ada-002: </div>${$('#embeddingAda').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">HuggingFace: </div>${$('#embeddingHuggingFace').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> - ${$('#embeddingHuggingFaceModel').val()}</li>
                        </ul>
                    </div>    
                    <div class="col-md-6 mt-3">
                        <p><strong>Vector DB:</strong> ${$('input[name="vectorDB"]:checked').attr('id')}</p>
                    </div>
                    <div class="col-md-6 mt-3">
                        <p><strong>Retriever:</strong></p>
                        <ul>
                            <li><div class="row"><div class="col-8">Vector DB - Similarity Search: </div>${$('#vectorSimilarity').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Vector DB - MMR: </div>${$('#vectorMMR').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">BM25 Retriever: </div>${$('#bm25Retriever').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Multi Query Retriever: </div>${$('#multiQuery').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Parent Doc Retriever - Full Documents: </div>${$('#parentDocFullDoc').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Parent Doc Retriever - Large Chunks: </div>${$('#parentDocLargeChunk').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                        </ul>
                    </div>    
                    <div class="col-md-6 mt-3">
                        <p><strong>Top k:</strong></p>
                        <ul>
                            <li><div class="row"><div class="col-3">5: </div>${$('#topK5').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-3">10: </div>${$('#topK10').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-3">20: </div>${$('#topK20').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                        </ul>
                    </div>    
                    <div class="col-md-6 mt-3">
                        <p><strong>Compression:</strong></p>
                        <ul>
                            <li><div class="row"><div class="col-8">Contextual Compression: </div>${$('#contextualCompression').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle mx-1 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <ul>
                                <li><div class="row"><div class="col-8">Long Context Reorder: </div>${$('#longContextReorder').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                                <li><div class="row"><div class="col-8">Cross Encoder Re-ranker: </div>${$('#crossEncoderReranker').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                                <li><div class="row"><div class="col-8">Embedding Redundant Filter: </div>${$('#embeddingsRedundantFilter').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                                <li><div class="row"><div class="col-8">Embedding Clustering Filter: </div>${$('#embeddingsClusteringFilter').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                                <li><div class="row"><div class="col-8">LLM Chain Filter: </div>${$('#llmChainFilter').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            </ul>        
                        </ul>
                    </div>    
                    <div class="col-md-6 mt-3">
                        <p><strong>LLM:</strong></p>
                        <ul>
                            <li><div class="row"><div class="col-3">GPT-3.5 Turbo: </div>${$('#gpt35').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-3">GPT-4o: </div>${$('#gpt4o').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-3">GPT-4 Turbo: </div>${$('#gpt4Turbo').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-3">HuggingFace: </div>${$('#llmHuggingFace').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> - ${$('#llmHuggingFaceModel').val()}</li>
                        </ul>
                    </div>
                </div>    
            `
        }

        // Fill the review section with selections from all steps
        const selections = `
            <div class="row row-cols-2">
                <div class="col-md-5"><strong>Description:</strong></div><div class="col-md-7">${$('#description').val()}</div>
                <div class="col-md-5"><strong>Source data:</strong></div><div class="col-md-7">${$('#sourceData').val()}</div>
                <div class="col-md-5"><strong>Use Pre-defined RAG Templates:</strong></div>
                <div class="col-md-7">${$('#compareTemplates').is(':checked')? '<i class="fas fa-check-circle me-2 text-success"></i>' : '<i class="fa-regular fa-circle me-2 text-secondary"></i>'}</div>
                <div class="col-md-5"><strong>Create Custom RAG Configurations:</strong></div>
                <div class="col-md-7">${$('#includeNonTemplated').is(':checked')? '<i class="fas fa-check-circle me-2 text-success"></i>' : '<i class="fa-regular fa-circle me-2 text-secondary"></i>'}</div>
            </div>
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
                min: chunk_size_values[0],
                max: chunk_size_values[1]
            },
            embeddingModel: {
                "OpenAI:text-embedding-3-small": $('#embeddingSmall').is(':checked'),
                "OpenAI:text-embedding-3-large": $('#embeddingLarge').is(':checked'),
                "OpenAI:text-embedding-ada-002": $('#embeddingAda').is(':checked'),
                "HuggingFace": $('#embeddingHuggingFace').is(':checked'),
                "AzureOAI": $('#embeddingAzureOAI').is(':checked'),
                "GoogleVertexAI": $('#embeddingGoogleVertexAI').is(':checked')
            },
            huggingfaceEmbeddingModel: $('#embeddingHuggingFace').is(':checked') ? 'HF:'+$('#embeddingHuggingFaceModel').val() : $('#embeddingHuggingFaceModel').val(),
            azureOAIEmbeddingModel: $('#embeddingAzureOAI').is(':checked') ? 'Azure:'+$('#embeddingAzureOAIModel').val() : $('#embeddingAzureOAIModel').val(),
            googleVertexAIEmbeddingModel: $('#embeddingGoogleVertexAI').is(':checked') ? 'GoogleVertexAI:'+$('#embeddingGoogleVertexAIModel').val() : $('#embeddingGoogleVertexAIModel').val(),
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
                "OpenAI:gpt-3.5-turbo": $('#gpt35').is(':checked'),
                "OpenAI:gpt-4o": $('#gpt4o').is(':checked'),
                "OpenAI:gpt-4-turbo": $('#gpt4Turbo').is(':checked'),
                "HuggingFace": $('#llmHuggingFace').is(':checked'),
                "Groq": $('#llmGroq').is(':checked'),
                "AzureOAI": $('#llmAzureOAI').is(':checked'),
                "GoogleVertexAI": $('#llmGoogleVertexAI').is(':checked')
            },
            huggingfaceLLMModel: $('#llmHuggingFace').is(':checked') ? 'HF:'+$('#llmHuggingFaceModel').val() : $('#llmHuggingFaceModel').val(),
            groqLLMModel: $('#llmGroq').is(':checked') ? 'Groq:'+$('#llmGroqModel').val() : $('#llmGroqModel').val(),
            azureOAILLMModel: $('#llmAzureOAI').is(':checked') ? 'Azure:'+$('#llmAzureOAIModel').val() : $('#llmAzureOAIModel').val(),
            googleVertexAILLMModel: $('#llmGoogleVertexAI').is(':checked') ? 'GoogleVertexAI:'+$('#llmGoogleVertexAIModel').val() : $('#llmGoogleVertexAIModel').val(),
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

        if (projectData.optimization === "bayesianOptimization") {
            projectData.numRuns = $('#numRuns').val();
        }
    
        console.log(JSON.stringify(projectData));

        $('#step4').hide();
        $('#progressSection').show();

        fetchLogUpdates();
        fetchProgressUpdates();

        $.ajax({
            type: "POST",
            url: "/rbuilder",
            contentType: "application/json",
            data: JSON.stringify(projectData),
            success: function(response) {
                if (response.status === "success") {
                    fetchLogUpdates();
                    fetchProgressUpdates();
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
                clearInterval(progressInterval);
                clearInterval(smoothInterval);
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

    let lastKnownRun = -1;
    let lastUpdateTime = Date.now();

    function fetchProgressUpdates() {
        const progressInterval = setInterval(function () {
            $.ajax({
                type: "GET",
                url: "/progress",
                success: function (response) {
                    console.log(JSON.stringify(response));
                    const currentRun = response.current_run;
                    const totalRuns = response.total_runs;

                    if (currentRun > lastKnownRun) {
                        lastKnownRun = currentRun;
                        const progressPercentage = Math.min((currentRun / totalRuns) * 100, 100);
                        console.log('ProgressPercentage: ', progressPercentage);
                        $('#progressText').text(`Running ${currentRun}/${totalRuns}...`);

                        if (smoothInterval) {
                            clearInterval(smoothInterval);
                        }
                        smoothProgressUpdate(progressPercentage, currentRun, totalRuns);
                        lastUpdateTime = Date.now();
                    }

                    if (Date.now() - lastUpdateTime > 300000) { // In case there's no progress for the last 120 secs since smoothProgressUpdate stopped (60 secs ago)
                        $('#progressText').text(`Running ${currentRun}/${totalRuns}... (Current run is taking longer than expected)`);
                    }

                    if (currentRun >= totalRuns) {
                        clearInterval(progressInterval);
                    }
                },
                error: function (error) {
                    console.error(error);
                }
            });

        }, 20000); // Update every 20 seconds
    }

    function smoothProgressUpdate(progressPercentage, currentRun, totalRuns) {
        const duration = 240 * 1000; // 60 seconds
        const interval = 2000; // 2 seconds
        const steps = duration / interval;
        const increment = (1 / totalRuns) * 100 / steps;
        console.log('currentRun: ', currentRun);
        console.log('totalRuns: ', totalRuns);
        console.log('duration: ', duration);
        console.log('interval: ', interval);
        console.log('steps: ', steps);
        console.log('Increment: ', increment);
        let currentProgress = progressPercentage;
        let targetProgress = Math.min(((currentRun + 1) / totalRuns) * 100, 100);

        smoothInterval = setInterval(function () {
            if (currentProgress >= targetProgress) {
                clearInterval(smoothInterval);
            } else {
                currentProgress += increment;
                console.log('currentProgress += increment: ', currentProgress);
                $('#progressBar').css('width', `${currentProgress - 0.1}%`).attr('aria-valuenow', currentProgress);
            }
        }, interval);
    }
    
    function fetchLogUpdates() {
        const logInterval = setInterval(function () {
            $.ajax({
                type: "GET",
                url: "/get_log_updates",
                success: function (response) {
                    $('#logOutput').text(response.log_content);
                    const logOutputElement = $('#logOutput');
                    logOutputElement.text(response.log_content);

                    // Automatically scroll to the bottom of the log output
                    logOutputElement.scrollTop(logOutputElement[0].scrollHeight);
                    
                    const logContent = response.log_content;
                    if (logContent.includes("Processing finished successfully.")) {
                        clearInterval(logInterval);
                        $('#progressSection').hide();
                        $('#completionSection').show();
                    }
                },
                error: function (error) {
                    console.error(error);
                }
            });
        }, 2000); // Update every 2 seconds
    }
});



