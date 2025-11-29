// AnalysisDto.cs - »—ѕ–ј¬Ћ≈ЌЌјя ¬≈–—»я
namespace BackendSite.Dtos
{
    public class AnalysisResultDto
    {
        public string comment { get; set; } = string.Empty;  // строчные буквы!
        public double class_label { get; set; }              // строчные буквы!
        public double probability { get; set; }
        public DateTime created_date { get; set; }
    }
}