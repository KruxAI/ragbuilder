def agg_eval_prompts(i):
    print("eval_prompts initiated")
    import pandas as pd
    input_csv_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_eval_results"+str(i)+".csv"
    results_df = pd.read_csv(input_csv_path)
    average_correctness = results_df.groupby('prompt_key')['answer_correctness'].mean().reset_index()
    average_correctness.columns = ['prompt_key', 'average_correctness']
    average_correctness.to_csv('rag_average_correctness'+str(i)+'.csv', index=False)
    print("The results have been saved to 'average_correctness.csv'")
    print("eval_prompts completed",i)

agg_eval_prompts(8)
