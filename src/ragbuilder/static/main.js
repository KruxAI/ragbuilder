
let progressInterval;
let smoothInterval;
let currentRunId = null;

function validateSourceData() {
    const sourceData = $('#sourceData').val();

    $.ajax({
        type: "POST",
        url: "/check_source_data",
        contentType: "application/json",
        data: JSON.stringify({ sourceData: sourceData }),
        success: function (response) {
            var sourceDataError = document.getElementById("sourceDataError")
            if (response.valid) {
                console.log('Source data is valid.');
                sourceDataError.innerHTML = "";
                $('#nextStep1').prop('disabled', false);
                updateDataSizeInfo(response.size, response.exceeds_threshold);
            } else {
                sourceDataError.innerHTML = "<span style='color: red;'>"+"Invalid source data. Please check the URL or file/ directory path.</span>";
                $('#nextStep1').prop('disabled', true);
                $('#dataSizeInfo').hide();
            }
        },
        error: function (error) {
            console.error('Error validating source data:', error);
        }
    });
}

function updateDataSizeInfo(size, exceedsThreshold) {
    const sizeInMB = size / (1024 * 1024);
    const sizeDisplay = sizeInMB >= 1024 ? `${(sizeInMB / 1024).toFixed(2)} GB` : `${sizeInMB.toFixed(2)} MB`;
    
    let infoText = '';
    if (exceedsThreshold) {
        infoText = `Your dataset is relatively large (${sizeDisplay}). RAGBuilder can sample your data to provide quicker initial results. You can always run the full analysis later if you're satisfied with the initial results.`;
        $('#useSampling').prop('checked', true);
    } else {
        infoText = `Your dataset size is ${sizeDisplay}. RAGBuilder can process this dataset without sampling.`;
        $('#useSampling').prop('checked', false);
    }

    $('#dataSizeInfo').text(infoText).show();
    $('#samplingOption').show();
    $('#dataPreProcessingOption').show();
}


function formatModelSelection(provider, model) {
    return `${provider}:${model}`;
}

function getModel(selectedID, modelName) {
    const elt = $('#'+selectedID)[0];
    const provider = $(elt.options[elt.selectedIndex]).closest('optgroup').prop('label');
    const model = $('#'+modelName).val();
    if (provider === 'OpenAI') {
        return formatModelSelection('OpenAI', $(elt.options[elt.selectedIndex]).val());
    } else {
        return formatModelSelection($(elt.options[elt.selectedIndex]).val(), model);
    }
}

function loadTemplates() {
    $.ajax({
        url: '/templates',
        type: 'GET',
        success: function(response) {
            console.log(response);
            let templatesHtml = '';
            response.templates.forEach(template => {
                templatesHtml += `
                    <div class="card mb-3">
                        <div class="card-body">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="${template.id}" id="template-${template.id}" name="templateCheckbox" checked>
                                <label class="form-check-label" for="template-${template.id}">
                                    <h6 class="mb-1">${template.name}</h6>
                                    <p class="text-muted small mb-0">${template.description}</p>
                                </label>
                            </div>
                        </div>
                    </div>
                `;
            });
            $('#templatesList').html(templatesHtml);
        },
        error: function(error) {
            console.error('Error loading templates:', error);
            $('#templatesList').html('<p class="text-danger">Error loading templates. Please try again.</p>');
        }
    });
}

function fetch_current_run_id() {
    $.ajax({
        type: "GET",
        url: "/get_current_run_id",
        success: function(response) {
            currentRunId = response.run_id;
        },
        error: function(error) {
            console.error("Failed to fetch current run ID:", error);
        }
    });
}

$(document).ready(function () {
    // Show modal to capture user inputs and send it to backend.
    $('#newProject').click(function () {
        $('#newProjectModal').modal('show')
    });
    var dataExists = "";
    var existingSynthDataPath = "";
    var chunk_size_values = [500, 2000]

    var collapseAdv = new bootstrap.Collapse(document.getElementById('collapseAdv'), {
        toggle: false
    });
    
    // Check if source data path is valid
    $('#sourceData').on('blur', function() {
        validateSourceData();
    });

    $('#useSampling').change(function() {
        if ($(this).is(':checked')) {
            $('#samplingInfo').show();
        } else {
            $('#samplingInfo').hide();
        }
    });

    document.querySelectorAll('[data-mdb-toggle="collapse"]').forEach(element => {
        new mdb.Collapse(element, {toggle: false});
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

    $('#sotaEmbeddingModel').change(function() {
        var selectedValue = $(this).val();
        if (['HF', 'AzureOAI', 'GoogleVertexAI', 'Ollama'].includes(selectedValue)) {
            $('#customSotaEmbeddingModel').show();
        } else {
            $('#customSotaEmbeddingModel').hide().val('');
        }
    });

    $('#sotaLLMModel').change(function() {
        var selectedValue = $(this).val();
        if (['HF', 'Groq', 'AzureOAI', 'GoogleVertexAI', 'Ollama'].includes(selectedValue)) {
            $('#customSotaLLMModel').show();
        } else {
            $('#customSotaLLMModel').hide().val('');
        }
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

    $('#embeddingOllama').change(function () {
        if (this.checked) {
            $('#embeddingOllamaModelDiv').show();
        } else {
            $('#embeddingOllamaModelDiv').hide();
            $('#embeddingOllamaModel').val('');
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

    $('#llmOllama').change(function () {
        if (this.checked) {
            $('#llmOllamaModelDiv').show();
        } else {
            $('#llmOllamaModelDiv').hide();
            $('#llmOllamaModel').val('');
        }
    });

    // $('#contextualCompression').change(function() {
    //     if (!this.checked) {
    //         $('#mxbai-rerank-base-v1, #mxbai-rerank-large-v1, #bge-reranker-base, #flashrank, #cohere, #jina, #colbert, #rankllm, #longContextReorder, #embeddingsRedundantFilter, #embeddingsClusteringFilter, #llmChainFilter').prop('checked', false).prop('disabled', true);
    //     } else {
    //         $('#mxbai-rerank-base-v1, #mxbai-rerank-large-v1, #bge-reranker-base, #flashrank, #cohere, #jina, #colbert, #rankllm, #longContextReorder, #embeddingsRedundantFilter, #embeddingsClusteringFilter, #llmChainFilter').prop('disabled', false);
    //     }
    // });
    // let lastState = {};

    // function updateLastState() {
    //     $('#compression-opts input[type="checkbox"]').not('#contextualCompression').each(function() {
    //         lastState[this.id] = this.checked;
    //     });
    // }

    // updateLastState();

    // $('#compression').change(function() {
    //     const isChecked = this.checked;
    //     $('#compression-opts input[type="checkbox"]').prop('disabled', !isChecked);
        
    //     if (isChecked) {
    //         $('#contextualCompression').prop('checked', true);
    //         $('#compression-opts input[type="checkbox"]').prop('checked', true);
    //         updateLastState();
    //     } else {
    //         $('#compression-opts input[type="checkbox"]').prop('checked', false);
    //         updateLastState();
    //     }
    // });

    // $('#contextualCompression').change(function() {
    //     const isChecked = this.checked;
    //     $('#compression-opts input[type="checkbox"]').not('#contextualCompression').each(function() {
    //         if (isChecked) {
    //             $(this).prop('disabled', false).prop('checked', lastState[this.id]);
    //         } else {
    //             lastState[this.id] = this.checked;
    //             $(this).prop('disabled', true).prop('checked', false);
    //         }
    //     });
    // });

    // Show or hide the number of runs input based on the selected optimization option
    $('input[name="optimization"]').change(function () {
        if ($('#bayesianOptimization').is(':checked')) {
            $('#numRunsContainer').show();
            $('#nJobsContainer').show();
        } else {
            $('#numRunsContainer').hide();
            $('#nJobsContainer').hide();
        }
    });

    $('#evalEmbedding').change(function() {
        var parent = $(this.options[this.selectedIndex]).closest('optgroup').prop('label');
        if (parent === 'OpenAI') {
            $('#customEvalEmbedding').hide().val('');
        } else {
            $('#customEvalEmbedding').show();
        }
    });
    
    $('#evalLLM').change(function() {
        var parent = $(this.options[this.selectedIndex]).closest('optgroup').prop('label');
        if (parent === 'OpenAI') {
            $('#customEvalLLM').hide().val('');
        } else {
            $('#customEvalLLM').show();
        }
    });

    $('#generatorEmbedding').change(function() {
        var parent = $(this.options[this.selectedIndex]).closest('optgroup').prop('label');
        if (parent === 'OpenAI') {
            $('#customGenEmbedding').hide().val('');
        } else {
            $('#customGenEmbedding').show();
        }
    });
    
    $('#criticLLM').change(function() {
        var parent = $(this.options[this.selectedIndex]).closest('optgroup').prop('label');
        if (parent === 'OpenAI') {
            $('#customCriticLLM').hide().val('');
        } else {
            $('#customCriticLLM').show();
        }
    });

    $('#generatorLLM').change(function() {
        var parent = $(this.options[this.selectedIndex]).closest('optgroup').prop('label');
        if (parent === 'OpenAI') {
            $('#customGenLLM').hide().val('');
        } else {
            $('#customGenLLM').show();
        }
    });

    $('#nextStep1').click(function () {
        const sourceData = $('#sourceData').val();
        const useSampling = $('#useSampling').is(':checked');
        $.ajax({
            type: "POST",
            url: "/check_test_data",
            contentType: "application/json",
            data: JSON.stringify({ sourceData: sourceData, useSampling: useSampling }),
            success: function (response) {
                if (response.exists) {
                    existingSynthDataPath = response.path;
                    dataExists=`<p><strong>Existing synthetic test data found for the provided source dataset.</strong><br>Path: ${response.path}</p>`
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

        if ($('#compareTemplates').is(':checked')) {
            $('#step1').hide();
            loadTemplates();
            $('#step1b').show();
        } else {
            $('#step1').hide();
            $('#step2').show();

        }
    });

    $('#nextStep1b').click(function () {
        if ($('#includeNonTemplated').is(':checked')) {
            $('#step1b').hide();
            $('#step2').show();
        } else {
            $('#step1b').hide();
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

        const evalFramework = $('#evalFramework').val();
        const evalEmbedding = getModel('evalEmbedding', 'customEvalEmbedding');
        const evalLLM = getModel('evalLLM', 'customEvalLLM');

        const evaluationHtml = `
            <p><strong>Evaluation Framework:</strong> ${evalFramework}</p>
            <p><strong>Evaluation Embedding Model:</strong> ${evalEmbedding}</p>
            <p><strong>Evaluation LLM:</strong> ${evalLLM}</p>
        `;

        testDataHtml += evaluationHtml;


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
                            <li><div class="row"><div class="col-8">HuggingFace: </div>${$('#embeddingHuggingFace').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> ${$('#embeddingHuggingFaceModel').val()}</li>
                            <li><div class="row"><div class="col-8">Azure OpenAI: </div>${$('#embeddingAzureOAI').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> ${$('#embeddingAzureOAIModel').val()}</li>
                            <li><div class="row"><div class="col-8">Google Vertex AI: </div>${$('#embeddingGoogleVertexAI').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> ${$('#embeddingGoogleVertexAIModel').val()}</li>
                            <li><div class="row"><div class="col-8">Ollama: </div>${$('#embeddingOllama').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> ${$('#embeddingOllamaModel').val()}</li>
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
                            <li><div class="row"><div class="col-8">Colbert Retriever: </div>${$('#colbertRetriever').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
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
                        <p><strong>Re-ranking/ Compression:</strong></p>
                        <ul>
                            <li><div class="row"><div class="col-8">Mixedbread-ai/mxbai-rerank-base-v1: </div>${$('#mxbai-rerank-base-v1').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Mixedbread-ai/mxbai-rerank-large-v1: </div>${$('#mxbai-rerank-large-v1').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">BAAI/bge-reranker-base: </div>${$('#bge-reranker-base').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Flashrank/ms-marco-MiniLM-L-12-v2: </div>${$('#flashrank').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Cohere/rerank-english-v3.0: </div>${$('#cohere').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Jina/jina-reranker-v1-base-en: </div>${$('#jina').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Colbert v2.0: </div>${$('#colbert').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Rankllm/gpt-4o: </div>${$('#rankllm').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Long Context Reorder: </div>${$('#longContextReorder').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Embedding Redundant Filter: </div>${$('#embeddingsRedundantFilter').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">Embedding Clustering Filter: </div>${$('#embeddingsClusteringFilter').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-8">LLM Chain Filter: </div>${$('#llmChainFilter').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                        </ul>
                    </div>    
                    <div class="col-md-6 mt-3">
                        <p><strong>LLM:</strong></p>
                        <ul>
                            <li><div class="row"><div class="col-6">GPT-4o mini: </div>${$('#gpt4oMini').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>    
                            <li><div class="row"><div class="col-6">GPT-4o: </div>${$('#gpt4o').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-6">GPT-3.5 Turbo: </div>${$('#gpt35').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-6">GPT-4 Turbo: </div>${$('#gpt4Turbo').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div></li>
                            <li><div class="row"><div class="col-6">HuggingFace: </div>${$('#llmHuggingFace').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> ${$('#llmHuggingFaceModel').val()}</li>
                            <li><div class="row"><div class="col-6">Groq: </div>${$('#llmGroq').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> ${$('#llmGroqModel').val()}</li>
                            <li><div class="row"><div class="col-6">Azure OpenAI: </div>${$('#llmAzureOAI').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> ${$('#llmAzureOAIModel').val()}</li>
                            <li><div class="row"><div class="col-6">Google Vertex AI: </div>${$('#llmGoogleVertexAI').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> ${$('#llmGoogleVertexAIModel').val()}</li>
                            <li><div class="row"><div class="col-6">Ollama: </div>${$('#llmOllama').is(':checked')? '<div class="col-1"><i class="fas fa-check-circle me-2 text-success"></i></div>' : '<div class="col-1"><i class="fa-regular fa-circle me-2 text-secondary"></i></div>'}</div> ${$('#llmOllamaModel').val()}</li>
                        </ul>
                    </div>
                </div>    
            `
        }

        // Fill the review section with selections from all steps
        const selections = `
            <div class="row row-cols-2">
                <div class="col-md-4"><strong>Description:</strong></div><div class="col-md-8">${$('#description').val()}</div>
                <div class="col-md-4"><strong>Source data:</strong></div><div class="col-md-8">${$('#sourceData').val()}</div>
                <div class="col-md-4"><strong>Use data sampling:</strong></div>
                <div class="col-md-8">${$('#useSampling').is(':checked')? '<i class="fas fa-check-circle me-2 text-success"></i>' : '<i class="fa-regular fa-circle me-2 text-secondary"></i>'}</div>
                <div class="col-md-4"><strong>Use Pre-defined RAG Templates:</strong></div>
                <div class="col-md-8">${$('#compareTemplates').is(':checked')? '<i class="fas fa-check-circle me-2 text-success"></i>' : '<i class="fa-regular fa-circle me-2 text-secondary"></i>'}</div>
                <div class="col-md-4"><strong>Create Custom RAG Configurations:</strong></div>
                <div class="col-md-8">${$('#includeNonTemplated').is(':checked')? '<i class="fas fa-check-circle me-2 text-success"></i>' : '<i class="fa-regular fa-circle me-2 text-secondary"></i>'}</div>
            </div>
            ${customSelections}
            ${testDataHtml}
        `;
        $('#step3').hide();
        $('#step4').show();
        $('#review').html(selections);
    });

    $('#previousStep1b').click(function () {
        $('#step1b').hide();
        $('#step1').show();
    });

    $('#previousStep2').click(function () {
        $('#step2').hide();
        if ($('#compareTemplates').is(':checked')) {
            $('#step1b').show();
        } else {
            $('#step1').show();
        }
    });

    $('#previousStep3').click(function () {
        $('#step3').hide();
        if ($('#includeNonTemplated').is(':checked')) {
            $('#step2').show();
        } else {
            if ($('#compareTemplates').is(':checked')) {
                $('#step1b').show();
            } else {
                $('#step1').show();
            }
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
        const dataProcessors = $('#data-preprocessing input[type="checkbox"]:checked')
            .map(function() {
                return this.id;
            })
            .get();

        const projectData = {
            description: $('#description').val(),
            sourceData: $('#sourceData').val(),
            useSampling: $('#useSampling').is(':checked'),
            compareTemplates: $('#compareTemplates').is(':checked'),
            includeNonTemplated: $('#includeNonTemplated').is(':checked'),
            selectedTemplates: [],
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
                "GoogleVertexAI": $('#embeddingGoogleVertexAI').is(':checked'),
                "Ollama": $('#embeddingOllama').is(':checked'),
            },
            huggingfaceEmbeddingModel: $('#embeddingHuggingFace').is(':checked') ? 'HF:'+$('#embeddingHuggingFaceModel').val() : $('#embeddingHuggingFaceModel').val(),
            azureOAIEmbeddingModel: $('#embeddingAzureOAI').is(':checked') ? 'AzureOAI:'+$('#embeddingAzureOAIModel').val() : $('#embeddingAzureOAIModel').val(),
            googleVertexAIEmbeddingModel: $('#embeddingGoogleVertexAI').is(':checked') ? 'GoogleVertexAI:'+$('#embeddingGoogleVertexAIModel').val() : $('#embeddingGoogleVertexAIModel').val(),
            ollamaEmbeddingModel: $('#embeddingOllama').is(':checked') ? 'Ollama:'+$('#embeddingOllamaModel').val() : $('#embeddingOllamaModel').val(),
            vectorDB: $('input[name="vectorDB"]:checked').attr('id'),
            retriever: {
                vectorSimilarity: $('#vectorSimilarity').is(':checked'),
                vectorMMR: $('#vectorMMR').is(':checked'),
                bm25Retriever: $('#bm25Retriever').is(':checked'),
                colbertRetriever: $('#colbertRetriever').is(':checked'),
                multiQuery: $('#multiQuery').is(':checked'),
                parentDocFullDoc: $('#parentDocFullDoc').is(':checked'),
                parentDocLargeChunk: $('#parentDocLargeChunk').is(':checked')
            },
            topK: {
                search_k_5: $('#topK5').is(':checked'),
                search_k_10: $('#topK10').is(':checked'),
                search_k_20: $('#topK20').is(':checked')
            },
            compressors: {
                "mixedbread-ai/mxbai-rerank-base-v1": $('#mxbai-rerank-base-v1').is(':checked'),
                "mixedbread-ai/mxbai-rerank-large-v1": $('#mxbai-rerank-large-v1').is(':checked'),
                "BAAI/bge-reranker-base": $('#bge-reranker-base').is(':checked'),
                "flashrank": $('#flashrank').is(':checked'),
                "cohere": $('#cohere').is(':checked'),
                "jina": $('#jina').is(':checked'),
                "colbert": $('#colbert').is(':checked'),
                "rankllm": $('#rankllm').is(':checked'),
                "LongContextReorder": $('#longContextReorder').is(':checked'),
                "EmbeddingsRedundantFilter": $('#embeddingsRedundantFilter').is(':checked'),
                "EmbeddingsClusteringFilter": $('#embeddingsClusteringFilter').is(':checked'),
                "LLMChainFilter": $('#llmChainFilter').is(':checked')
            },
            llm: {
                "OpenAI:gpt-4o-mini": $('#gpt4oMini').is(':checked'),
                "OpenAI:gpt-4o": $('#gpt4o').is(':checked'),
                "OpenAI:gpt-3.5-turbo": $('#gpt35').is(':checked'),
                "OpenAI:gpt-4-turbo": $('#gpt4Turbo').is(':checked'),
                "HuggingFace": $('#llmHuggingFace').is(':checked'),
                "Groq": $('#llmGroq').is(':checked'),
                "AzureOAI": $('#llmAzureOAI').is(':checked'),
                "GoogleVertexAI": $('#llmGoogleVertexAI').is(':checked'),
                "Ollama": $('#llmOllama').is(':checked'),
            },
            huggingfaceLLMModel: $('#llmHuggingFace').is(':checked') ? 'HF:'+$('#llmHuggingFaceModel').val() : $('#llmHuggingFaceModel').val(),
            groqLLMModel: $('#llmGroq').is(':checked') ? 'Groq:'+$('#llmGroqModel').val() : $('#llmGroqModel').val(),
            azureOAILLMModel: $('#llmAzureOAI').is(':checked') ? 'AzureOAI:'+$('#llmAzureOAIModel').val() : $('#llmAzureOAIModel').val(),
            googleVertexAILLMModel: $('#llmGoogleVertexAI').is(':checked') ? 'GoogleVertexAI:'+$('#llmGoogleVertexAIModel').val() : $('#llmGoogleVertexAIModel').val(),
            ollamaLLMModel: $('#llmOllama').is(':checked') ? 'Ollama:'+$('#llmOllamaModel').val() : $('#llmOllamaModel').val(),
            generateSyntheticData: $('#generateSynthetic').is(':checked'),
            evalFramework: $('#evalFramework').val(),
            evalEmbedding: getModel('evalEmbedding', 'customEvalEmbedding'),
            evalLLM: getModel('evalLLM', 'customEvalLLM'),
            optimization: $('input[name="optimization"]:checked').attr('id')
        };

        projectData.dataProcessors = dataProcessors;

        if ($('#compareTemplates').is(':checked')) {
            $('input[name="templateCheckbox"]:checked').each(function() {
                projectData.selectedTemplates.push($(this).val());
            });

            projectData.sotaEmbeddingModel =  getModel('sotaEmbeddingModel', 'customSotaEmbeddingModel');
            projectData.sotaLLMModel =  getModel('sotaLLMModel', 'customSotaLLMModel');
        }

        if ($('#existingTestData').is(':checked')) {
            projectData.testDataPath = $('#testDataPath').val();
        } else if ($('#useExistingSynthData').is(':checked')) {
            projectData.existingSynthDataPath=`${existingSynthDataPath}`;
        } else {
            projectData.syntheticDataGeneration = {
                testSize: $('#testSize').val(),
                criticLLM: getModel('criticLLM', 'customCriticLLM'),
                generatorLLM: getModel('generatorLLM', 'customGenLLM'),
                generatorEmbedding: getModel('generatorEmbedding', 'customGenEmbedding')
            };
        }

        if (projectData.optimization === "bayesianOptimization") {
            projectData.numRuns = $('#numRuns').val();
            projectData.nJobs = $('#nJobs').val();
        }
    
        console.log(JSON.stringify(projectData));

        $('#step4').hide();
        $('#progressSection').show();
        $('#viewResultsBtn').hide();

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

    $('#viewResultsBtn').click(function() {
        if (currentRunId) {
            window.open(`/summary/${currentRunId}`, '_blank');
        } else {
            console.error('No run ID available');
            alert('Unable to view results: No active run ID found.');
        }
    });

    let lastKnownRun = -1;
    let lastUpdateTime = Date.now();

    function checkFirstEvalComplete(response) {
        if (response.first_eval_complete && !$('#viewResultsBtn').is(':visible')) {
            fetch_current_run_id();
            $('#viewResultsBtn').show();
            $('#viewResultsText').text('First evaluation complete. You can now view current results.');
        }
    }

    function fetchProgressUpdates() {
        const progressInterval = setInterval(function () {
            $.ajax({
                type: "GET",
                url: "/progress",
                success: function (response) {
                    const { current_run, total_runs, synth_data_gen_in_progress, first_eval_complete } = response;
    
                    if (synth_data_gen_in_progress === 1) {
                        handleSynthDataGeneration();
                    } else {
                        handleNormalProgress(current_run, total_runs);
                    }

                    checkFirstEvalComplete(response);
    
                    if (current_run >= total_runs && synth_data_gen_in_progress === 0) {
                        clearInterval(progressInterval);
                    }
                },
                error: function (error) {
                    console.error(error);
                }
            });
        }, 5000); // Update every 5 seconds
    }

    function handleSynthDataGeneration() {
        $('#progressText').text("Generating synthetic test data... (this may take a while)");
        if (!smoothInterval) {
            startSlowProgressBar();
        }
    }

    function handleNormalProgress(currentRun, totalRuns) {
        if (currentRun > lastKnownRun) {
            lastKnownRun = currentRun;
            const progressPercentage = Math.min((currentRun / totalRuns) * 100, 100);
            $('#progressText').text(`Running ${currentRun}/${totalRuns}...`);
    
            if (smoothInterval) {
                clearInterval(smoothInterval);
            }
            smoothProgressUpdate(progressPercentage, currentRun, totalRuns);
            lastUpdateTime = Date.now();
        }
    
        if (Date.now() - lastUpdateTime > 300000) {
            $('#progressText').text(`Running ${currentRun}/${totalRuns}... (Current run is taking longer than expected)`);
        }
    }
    
    function startSlowProgressBar() {
        let progress = 0;
        smoothInterval = setInterval(function () {
            progress += 0.1;
            if (progress > 100) {
                progress = 0;
            }
            $('#progressBar').css('width', `${progress}%`).attr('aria-valuenow', progress);
        }, 2000);
    }

    function smoothProgressUpdate(progressPercentage, currentRun, totalRuns) {
        const duration = 240 * 1000; // 60 seconds
        const interval = 2000; // 2 seconds
        const steps = duration / interval;
        const increment = (1 / totalRuns) * 100 / steps;
        // console.log('currentRun: ', currentRun);
        // console.log('totalRuns: ', totalRuns);
        // console.log('duration: ', duration);
        // console.log('interval: ', interval);
        // console.log('steps: ', steps);
        // console.log('Increment: ', increment);
        let currentProgress = progressPercentage;
        let targetProgress = Math.min(((currentRun + 1) / totalRuns) * 100, 100);

        smoothInterval = setInterval(function () {
            if (currentProgress >= targetProgress) {
                clearInterval(smoothInterval);
            } else {
                currentProgress += increment;
                // console.log('currentProgress += increment: ', currentProgress);
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
