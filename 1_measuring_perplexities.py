#!/usr/bin/env python3
import ast
import tiktoken
from model import GPTConfig, GPT
import os
import re
import numpy as np
import torch
import pandas as pd
import argparse
from contextlib import contextmanager


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

device = 'cuda'

prefixes = ["imo ", "to be fair, ", "exactly! ", "i think ", "i guess ", "i'm sure ", "hi! ", "hello! ", "i mean ", "ah ok, ", "ha! ", "well, ", "wellokay, ", "okay, ", "oh ", "haha true. ", "yeah. ", "um... ",
"you are right ", "to put it short ", "hmmm ", "oh oh ", "hehe, ",
"edit: ", "edited: ", "to reiterate, ", "so, ", "i know. ", "ah ok. ",
"i agree. ", "not gonna lie. "]

our_sentences = [
#". In a year, interest rates will be higher.",
#". In a year, inflation will be higher.",
#". In a year, unemployment will be higher.", 

". I believe that interest rates will go up in a year.",
". I think that interest rates will go up in 12 months.",
". I reckon interest rates will increase in a year.",
". Interest rate will be higher in a year's time.",
". A rise in interest rates will happen in a year.",
". Next year interest rate will be higher than now.",
". Interest rate for borrowing money will go up.",
". Over the course of 12 months, interest rate will grow.",
". Soon enough, interest rate will go up.",
". I heard interest rates will not go down any time soon.",
". Some say interest rate is going up and this won't change soon.",
". Higher interest rate is expected next year.",
". In 12 months interest rates will be high.",
". Interest rates are bound to be higher next year.",
". The Fed will raise interest rates next year.",
". I’m positive that Powell will continue in interest rate hikes.",
". Interest rates will go up after the next Fed meeting.",
". The cost of borrowing will increase next year.",
". Interest rates will increase after the next Fed meeting.",
". My take is that the price of borrowing will be higher next year.",


". Everything’s gonna be more expensive next year.",
". Prices are going to rise.",
". Bidenflation will continue.",
". Money will continue losing its value next year.",
". Retailers will hike up their prices next year.",
". Prices will go up.",
". Inflation will be higher.",
". Inflation will rise next year.",
". Prices of common goods will rise over the course of this year.",
". I expect everything to be more expensive in 12 months.", 
". Inflation will end up higher than it is now.",
". Inflation will probably keep rising well into next year.",
". Next year inflation will continue to rise.",
". In the upcoming year inflation will hit us hard.",
". Prices of goods are going to be higher.", 
". Inflation will spike next year.",
". Inflation will trend upwards next year.",
". Next year’s inflation will be bigger than this year’s.",
". Inflation is going to be worse.",
". Prices of food and goods are bound to go higher.",

". People will lose their jobs.",
". The labor market situation will be worse.",
". Everyone will lose their occupation over the next few months.",
". Next year, unemployment will increase.",
".  Unemployment will go up next year.",
". There’ll be a lot of layoffs.",
". Layoff season is coming next year.",
". I predict a lot of layoffs.",
". Layoffs will continue next year.",
". Big tech layoffs will spread to other sectors next year.",
". There’s gonna be even more unemployment next year.",
". Finding a job will become harder next year.",
". A surge in unemployment seems imminent within a year.",
". The next year could see a worrying rise in unemployment.",
". Unemployment forecasts for the next year are not looking good.",
". There's a trend suggesting rising unemployment in the upcoming months.",
". Jobless rates are set to climb in the coming year.",
". I bet unemployment numbers are going to spike soon.",
". Does anyone else think we’re heading for higher unemployment rates in the near future?",
". I wouldn't be surprised if we see a jump in unemployment stats in the next 12 months.",

#". Interest rate will go up in a year.",
#". Interest rate will go down in a year.",

#". Inflation will go up in a year.",
#". Inflation will go down in a year.",

#". Unemployment will go up in a year.",
#". Unemployment will go down in a year."

]
class GPTPerplexityAnalyzer:

    dtype = 'bfloat16'  # if torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = True  # use PyTorch 2.0 to compile the model to be faster
    exec(open('configurator.py').read())  # overrides from command line or config file
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    ctx = torch.no_grad()
    def __init__(self, root_dir, csv_file_path):
        self.root_dir = root_dir
        self.csv_file_path = csv_file_path
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[GPTPerplexityAnalyzer.dtype]
        self.ctx = torch.no_grad()
        self.enc = tiktoken.get_encoding("gpt2")  # Tiktoken is a faster BPE tokeniser than HF's
        self.ends_with = '. '
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        self.device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.


    def load_model(self, out_dir):
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)

        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)

        adjusted_state_dict = {}
        for k, v in checkpoint['model'].items():
            new_key = k.replace('_orig_mod.', '')

            
            adjusted_state_dict[new_key] = v

        model.load_state_dict(adjusted_state_dict, strict=False)

        model.eval()
        model.to(device)

        return model


    def load_data(self):
        df = pd.read_csv(self.csv_file_path, delimiter='\t', header=None)
        context = df.iloc[1, 1]  
        parsed_df = pd.read_csv(self.csv_file_path, delimiter='\t', skiprows=2)
        return parsed_df, context

    def get_ppl(self, model, start, context):
        
        enc = tiktoken.get_encoding("gpt2")  # Tiktoken is a faster BPE tokeniser than HF's
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        start_ids = encode(context + ' ' + start + self.ends_with)
        x = (torch.tensor(start_ids[:-1], dtype=torch.long, device=self.device)[None, ...])
        y = (torch.tensor(start_ids[1:], dtype=torch.long, device=self.device)[None, ...])

        with torch.no_grad():
            with self.ctx:
                logits, loss = model(x, y)
                return loss.exp().item()

    def get_monthly_model_dirs(self):
        month_dirs = [os.path.join(self.root_dir, dir) for dir in os.listdir(self.root_dir) if re.match(r"^\d{4}-\d{2}$", dir)]
        if len(month_dirs) % 12 != 0:
            raise ValueError(f"The number of directories ({len(month_dirs)}) is not a multiple of 12.")
        return sorted(month_dirs)


    def aggregate_statistics_across_years(self, yearly_stats):
        data = []
        for year, months in yearly_stats.items():
            for month, values in months.items():
                if '-' in month:
                    numerical_month = month.split("-")[1]
                else:
                    print(f"Skipping entry: {year}, {month}")
                    continue

                for value in values:
                    data.append({'Year': year, 'Month': numerical_month, 'Perplexity': value})

        if not data:
            print("There is no data available in the expected format. :(")
            return None, None

        multi_level_df = pd.DataFrame(data)

        multi_level_df['Month'] = pd.to_numeric(multi_level_df['Month'])

        multi_level_df['Month'] = pd.to_datetime(multi_level_df['Month'], format='%m')

        avg_stats = multi_level_df.groupby('Month').mean()
        stddev_stats = multi_level_df.groupby('Month').std()

        return avg_stats, stddev_stats

    def segment_models_by_year(self):
        all_dirs = self.get_monthly_model_dirs()

        grouped_by_year = {}
        for dir in all_dirs:
            year = dir.split("-")[0]
            if year not in grouped_by_year:
                grouped_by_year[year] = []
            grouped_by_year[year].append(dir)

        return list(grouped_by_year.values())

    def compute_perplexities_by_year(self, weather_df, context, debug_mode=False):
        yearly_model_segments = self.segment_models_by_year()
        all_yearly_results = {}
        for year_segment in yearly_model_segments:
            yearly_results = {}
            for month_dir in year_segment:
                month_name = os.path.basename(month_dir) 
                if debug_mode:
                    print(f"Loading model from directory: {month_dir}")

                model = self.load_model(month_dir)
                monthly_results = []

                for index, row in weather_df.iterrows():
                    sentences = [row['Sentence']] + [row[f'Paraphrase {i}'] for i in range(1, 5) if
                                                     pd.notna(row[f'Paraphrase {i}'])]
                    avg_perplexity = np.mean([self.get_ppl(model, sentence, context) for sentence in sentences])
                    monthly_results.append(avg_perplexity)
                yearly_results[month_name] = monthly_results  

            year_name = os.path.basename(year_segment[0]).split("-")[0]
            all_yearly_results[year_name] = yearly_results
        return all_yearly_results

    def calculate_conditional_values(self, row, original_df):
        if pd.isna(row['stdev']) or pd.isna(row['average']) or pd.isna(row['coefficient_of_variation']):
            return pd.Series({
                'conditional_stdev': None,
                'conditional_average': None,
                'conditional_coefficient_of_variation': None,
                'conditional_min': None,
                'conditional_max': None,
                'conditional_Q1': None,
                'conditional_Q3': None
            })

        non_data_columns = ['stdev', 'average', 'coefficient_of_variation']

        if row['Stability'] == 1:
            return pd.Series({
                'conditional_stdev': row['stdev'],
                'conditional_average': row['average'],
                'conditional_coefficient_of_variation': row['coefficient_of_variation'],
                'conditional_min': row['min'],
                'conditional_max': row['max'],
                'conditional_Q1': row['Q1'],
                'conditional_Q3': row['Q3']
            })
        else:
            excluded_columns = [f"{year}-{str(month).zfill(2)}" for year in original_df.columns.str[:4].unique() for month
                                in row['Peak Months']]
            data_columns = original_df.columns.difference(excluded_columns + non_data_columns)
            included_data = original_df.loc[row.name, data_columns]
            if included_data.empty:
                return pd.Series({
                    'conditional_stdev': None,
                    'conditional_average': None,
                    'conditional_coefficient_of_variation': None,
                    'conditional_min': None,
                    'conditional_max': None,
                    'conditional_Q1': None,
                    'conditional_Q3': None
                })
            else:
                conditional_stdev = included_data.std()
                conditional_average = included_data.mean()
                conditional_coefficient_of_variation = conditional_stdev / conditional_average if conditional_average != 0 else np.nan
                conditional_min = included_data.min()
                conditional_max = included_data.max()
                conditional_Q1 = included_data.quantile(0.25)
                conditional_Q3 = included_data.quantile(0.75)

            return pd.Series({
                'conditional_stdev': conditional_stdev,
                'conditional_average': conditional_average,
                'conditional_coefficient_of_variation': conditional_coefficient_of_variation,
                'conditional_min': conditional_min,
                'conditional_max': conditional_max,
                'conditional_Q1': conditional_Q1,
                'conditional_Q3': conditional_Q3
            })
    

   

    def calculate_sentence_perplexities(self, model, context, sentence):
        perplexity = self.get_ppl(model, sentence, context)
        formatted_perplexity = "{:.2f}".format(perplexity)  
        return formatted_perplexity


            def main(self, sentences):
        model_dirs = self.get_monthly_model_dirs()
        results_df = pd.DataFrame(columns=[os.path.basename(dir) for dir in model_dirs])

        for sentence in sentences:
            print("Processing sentence: ", sentence)
            sentence_results = []
            for model_dir in model_dirs:
                model = self.load_model(model_dir)
                perplexity = self.calculate_sentence_perplexities(model, model_dir, sentence)
                formatted_perplexity = f"{perplexity:,.2f}"
                sentence_results.append(formatted_perplexity)
            print("Sentence results are: ", sentence_results)
            results_df.loc[len(results_df)] = sentence_results

        results_df.to_csv(self.output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <model_base_dir> <output_csv_file>")
        sys.exit(1)

    model_base_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    analyzer = GPTPerplexityAnalyzer(model_base_dir, output_file)
    analyzer.main(our_sentences)

    
