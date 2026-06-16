Работа сравнивает пять методов выравнивания больших языковых
моделей на единой базовой модели и едином датасете:
 
    SFT  — Supervised Fine-Tuning
    DPO  — Direct Preference Optimization
    IPO  — Identity Preference Optimization
    KTO  — Kahneman-Tversky Optimization
    APO  — Anchored Preference Optimization
 
Базовая модель:  Qwen/Qwen2.5-1.5B-Instruct
Дообучение:      QLoRA (4-bit NF4, LoRA r=16, alpha=32)
Датасет:         HuggingFaceH4/ultrafeedback_binarized
                 (SFT — train_sft; DPO/IPO/APO — train_prefs;
                  KTO — train_prefs с конвертацией chosen/rejected в
                  label=True/False)
Среда обучения:  Google Colab (GPU T4)
 
Preference-методы (DPO, IPO, APO, KTO) дообучались не от базовой
модели, а от уже обученного SFT-адаптера. DPO, IPO и APO используют один и
тот же DPOTrainer и различаются только параметром loss_type.
 
Оценка проводилась по трём метрикам: перплексия (PPL), оценки
LLM-as-a-Judge (модель-судья Qwen2.5-7B-Instruct) и Win-Rate.

ССЫЛКА НА GOOGLE DRIVE:
[https://drive.google.com/drive/folders/1S8TBEf4MIq5mOTXRvVrcElcQMfsvOGsq?usp=drive_link]
 
В нем лежит: 
Обученные QLoRA-адаптеры (по одной папке на метод):
 
    sft/    — адаптер SFT
    dpo/    — адаптер DPO
    ipo/    — адаптер IPO
    kto/    — адаптер KTO
    apo/    — адаптер APO
 
В каждой папке лежит LoRA-адаптер, а НЕ полная модель. Для запуска нужна
базовая модель Qwen2.5-1.5B-Instruct, поверх которой подключается адаптер.
 
Результаты экспериментов:
 
    final_metrics.json    — все итоговые метрики (training_metrics,
                            perplexity, judge_scores, win_rates)
    results_summary.csv   — сводная таблица: PPL, Helpful, Coherence,
                            Overall, Win-Rate с 95% ДИ
    comparison_results.png — сводное сравнение методов по всем метрикам
    method_profiles.png    — профиль качества по каждому методу
                            (PPL + оценки судьи + Win-Rate)
    training_curves.png    — кривые обучения (loss по шагам)
    winrate_ci_scatter.png — Win-Rate с доверительными интервалами
