import os 
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
def generate_feedback(metrics):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #overall rep data
    rep_count = metrics["total_reps"]
    reps = metrics["reps"]
    bar_devs = metrics["bar_path_dev"]
    tempos = metrics["tempo_per_rep"]

    #summary stats
    avg_bar_dev = np.nanmean(bar_devs)
    avg_tempo = np.mean(tempos)

    rep_metrics = []
    for i, rep in enumerate(reps):
        dev = bar_devs[i]
        tempo = tempos[i] if i < len(tempos) else np.nan
        
        bar_dev_str = f"{dev}px" if not np.isnan(dev) else "N/A"
        tempo_str = f"{tempo}s" if not np.isnan(tempo) else "N/A"

        rep_metrics.append(
            f"Rep {rep['rep_count']}: "
            f"Depth={rep['depth']}, "
            f"Bottom Angle={rep['bottom_angle']} degrees, "
            f"Bar Path Deviation={bar_dev_str}, "
            f"Tempo={tempo_str}"
        )
    breakdown = "\n".join(rep_metrics)
    
    #hotencode depth
    depth_encode = {"below" : 0, "parallel" : 0 , "partial" : 0}
    for rep in reps:
        depth_encode[rep["depth"]] += 1

    prompt = f"""You are an expert in squatting and you are analyzing someones training set. 
        Provide specific, actionable feedback.
    
    Total reps : {rep_count},
    Depth quality : {depth_encode["below"]} below parallel, 
        {depth_encode["parallel"]} parallel, 
        {depth_encode["partial"]} partial
    Average bar path devaition: {avg_bar_dev} this is in pixels, anything lower than 30px is fine.
    Average tempo: {avg_tempo} seconds per rep.
    Make sure the feedback is specific and focus on cues that could help the person squatting. 
    """
    response = client.chat.completions.create(
        model = "gpt-4o-mini", messages = [{"role": "user", "content" : prompt}], max_tokens=300, temperature=0.7)
    return response.choices[0].message.content




