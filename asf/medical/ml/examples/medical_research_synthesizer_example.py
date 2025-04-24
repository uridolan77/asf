"""
Medical Research Synthesizer Example

This script demonstrates how to use the Medical Research Synthesizer to process
and analyze medical research papers.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from asf.medical.ml.document_processing import MedicalResearchSynthesizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample medical text for demonstration
SAMPLE_TEXT = """
Title: Effects of Exercise on Glycemic Control in Type 2 Diabetes

Abstract:
Regular physical activity is a cornerstone in the management of type 2 diabetes mellitus (T2DM). This study investigated the effects of different exercise modalities on glycemic control in patients with T2DM. We conducted a 12-week randomized controlled trial comparing aerobic exercise, resistance training, and combined training. The primary outcome was change in HbA1c. Secondary outcomes included fasting glucose, insulin sensitivity, and quality of life measures.

Introduction:
Type 2 diabetes mellitus (T2DM) is a metabolic disorder characterized by hyperglycemia resulting from insulin resistance and relative insulin deficiency. The global prevalence of T2DM continues to rise, with an estimated 463 million adults living with diabetes in 2019, projected to reach 700 million by 2045. Physical activity is widely recommended as a key component in the management of T2DM, alongside medication and dietary modifications.

Previous studies have demonstrated that exercise improves glycemic control, reduces cardiovascular risk factors, and enhances overall well-being in individuals with T2DM. However, the comparative effectiveness of different exercise modalities remains incompletely understood. This study aimed to address this knowledge gap by directly comparing the effects of aerobic exercise, resistance training, and combined training on glycemic control and other health outcomes in patients with T2DM.

Methods:
Study Design and Participants:
We conducted a 12-week randomized controlled trial at the Diabetes Research Center. Adults aged 40-70 years with diagnosed T2DM (HbA1c 7.0-10.0%) were recruited. Exclusion criteria included insulin therapy, uncontrolled hypertension, cardiovascular disease, and contraindications to exercise. The study was approved by the Institutional Review Board, and all participants provided written informed consent.

Participants were randomly assigned to one of four groups: aerobic exercise (AE), resistance training (RT), combined aerobic and resistance training (CT), or control (C). Randomization was performed using a computer-generated sequence with stratification by gender and baseline HbA1c.

Exercise Interventions:
All exercise sessions were supervised by certified exercise physiologists. Participants in the intervention groups attended three sessions per week for 12 weeks.

The AE group performed 50 minutes of moderate-intensity aerobic exercise (60-70% of maximum heart rate) on treadmills or stationary bicycles. The RT group performed resistance exercises targeting major muscle groups using weight machines and free weights, with 3 sets of 10-12 repetitions at 60-80% of one-repetition maximum. The CT group performed 25 minutes of aerobic exercise followed by a shortened resistance training protocol. The control group was asked to maintain their usual activities and received general advice about physical activity.

Outcome Measures:
The primary outcome was change in HbA1c from baseline to 12 weeks. Secondary outcomes included fasting plasma glucose, insulin sensitivity (assessed by HOMA-IR), body composition (assessed by dual-energy X-ray absorptiometry), blood pressure, lipid profile, and quality of life (assessed by the SF-36 questionnaire).

Assessments were performed at baseline and after the 12-week intervention period. Blood samples were collected after an overnight fast.

Statistical Analysis:
Sample size was calculated to detect a difference of 0.5% in HbA1c between groups with 80% power and a significance level of 0.05. Accounting for an anticipated dropout rate of 15%, we aimed to recruit 120 participants (30 per group).

Data were analyzed using intention-to-treat principles. Between-group differences were assessed using analysis of covariance (ANCOVA) with baseline values as covariates. Pairwise comparisons with Bonferroni correction were performed for significant overall effects. Statistical analyses were performed using SPSS version 25.0.

Results:
Participant Characteristics:
A total of 124 participants were randomized, with 112 (90.3%) completing the study. Baseline characteristics were similar across groups, with a mean age of 58.4 ± 7.2 years, diabetes duration of 7.3 ± 4.1 years, and HbA1c of 7.8 ± 0.8%.

Primary Outcome:
All exercise groups showed significant reductions in HbA1c compared to the control group. The mean changes in HbA1c were -0.51% (95% CI: -0.73 to -0.29) in the AE group, -0.38% (95% CI: -0.60 to -0.16) in the RT group, and -0.73% (95% CI: -0.95 to -0.51) in the CT group, compared to -0.05% (95% CI: -0.27 to 0.17) in the control group (p<0.001 for between-group differences). The CT group showed significantly greater reductions in HbA1c compared to both the AE group (p=0.04) and the RT group (p=0.007).

Secondary Outcomes:
Fasting plasma glucose decreased significantly in all exercise groups compared to the control group, with the largest reduction in the CT group (-28.4 mg/dL, p<0.001). Insulin sensitivity improved in all exercise groups, with HOMA-IR decreasing by 1.52, 1.18, and 1.89 in the AE, RT, and CT groups, respectively (all p<0.01 vs. control).

Body composition improved in all exercise groups, with the RT and CT groups showing greater increases in lean mass compared to the AE group. Blood pressure and lipid profiles improved in all exercise groups, with no significant differences between exercise modalities. Quality of life scores improved significantly in all exercise groups compared to the control group, with the largest improvements in the CT group.

Adverse Events:
No serious adverse events related to the interventions were reported. Minor musculoskeletal discomfort was reported by 15% of participants in the RT group, 12% in the CT group, and 7% in the AE group, but these did not lead to discontinuation of the intervention.

Discussion:
This randomized controlled trial demonstrated that all three exercise modalities—aerobic, resistance, and combined training—significantly improved glycemic control in patients with T2DM compared to a control group. However, combined training provided the greatest benefits for HbA1c reduction and improvements in several secondary outcomes.

Our findings are consistent with previous studies showing the benefits of exercise for glycemic control in T2DM. The American Diabetes Association recommends both aerobic and resistance exercise for individuals with T2DM, and our results support this recommendation. The greater effectiveness of combined training may be attributed to the complementary physiological adaptations induced by aerobic and resistance exercise.

Aerobic exercise primarily enhances insulin sensitivity through improved cardiovascular function, increased muscle capillary density, and enhanced glucose transport. Resistance training increases muscle mass and strength, enhancing glucose disposal capacity and basal metabolic rate. When combined, these adaptations may have synergistic effects on glucose metabolism.

The magnitude of HbA1c reduction observed in our study, particularly in the combined training group (-0.73%), is clinically significant. Previous research has shown that each 1% reduction in HbA1c is associated with a 21% reduction in diabetes-related deaths and a 37% reduction in microvascular complications.

Limitations of our study include the relatively short intervention period (12 weeks) and the supervised nature of the exercise sessions, which may not reflect real-world adherence to exercise recommendations. Additionally, our participants had relatively well-controlled diabetes at baseline, and the effects might differ in those with poorer glycemic control.

Conclusion:
In conclusion, this study demonstrates that aerobic exercise, resistance training, and combined training all improve glycemic control in patients with T2DM, with combined training showing the greatest benefits. These findings support current recommendations for incorporating both aerobic and resistance exercise in the management of T2DM. Future research should focus on strategies to enhance long-term adherence to exercise programs and investigate the effects of different exercise prescriptions in diverse diabetes populations.

References:
1. American Diabetes Association. Standards of Medical Care in Diabetes-2022. Diabetes Care. 2022;45(Suppl 1):S1-S264.
2. Colberg SR, Sigal RJ, Yardley JE, et al. Physical Activity/Exercise and Diabetes: A Position Statement of the American Diabetes Association. Diabetes Care. 2016;39(11):2065-2079.
3. Umpierre D, Ribeiro PA, Kramer CK, et al. Physical activity advice only or structured exercise training and association with HbA1c levels in type 2 diabetes: a systematic review and meta-analysis. JAMA. 2011;305(17):1790-1799.
"""

def main():
    """Run the Medical Research Synthesizer example."""
    logger.info("Initializing Medical Research Synthesizer...")
    
    # Initialize the synthesizer with default settings
    synthesizer = MedicalResearchSynthesizer()
    
    # Process the sample text
    logger.info("Processing sample medical research paper...")
    result = synthesizer.process(SAMPLE_TEXT)
    
    # Print basic information
    print("\n" + "="*80)
    print(f"Title: {result.title}")
    print(f"Sections: {len(result.sections)}")
    print(f"Entities: {len(result.entities)}")
    print(f"Relations: {len(result.relations)}")
    print("="*80 + "\n")
    
    # Print summary
    if result.summary:
        print("RESEARCH SUMMARY:")
        print("-"*80)
        
        if result.summary.get('abstract'):
            print(f"Abstract: {result.summary['abstract']}\n")
        
        if result.summary.get('key_findings'):
            print("Key Findings:")
            for i, finding in enumerate(result.summary['key_findings'], 1):
                print(f"  {i}. {finding}")
            print()
        
        if result.summary.get('clinical_implications'):
            print("Clinical Implications:")
            for i, implication in enumerate(result.summary['clinical_implications'], 1):
                print(f"  {i}. {implication}")
            print()
    
    # Print some extracted entities
    if result.entities:
        print("SAMPLE ENTITIES:")
        print("-"*80)
        for i, entity in enumerate(result.entities[:10], 1):
            print(f"{i}. {entity.text} ({entity.label})")
            if entity.cui:
                print(f"   UMLS CUI: {entity.cui}")
        print()
    
    # Print some extracted relations
    if result.relations:
        print("SAMPLE RELATIONS:")
        print("-"*80)
        for i, relation in enumerate(result.relations[:10], 1):
            print(f"{i}. {relation['head_entity']} --[{relation['relation_type']}]--> {relation['tail_entity']}")
        print()
    
    # Save results to output directory
    output_dir = "output"
    logger.info(f"Saving results to {output_dir}...")
    synthesizer.save_results(result, output_dir)
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main()
