namespace BackendSite.Dtos
{
    public class AnalysisResultDto
    {
        public string Comment { get; set; } = string.Empty;
        public double ClassLabel { get; set; }
        public double Probability { get; set; }
        public DateTime CreatedDate { get; set; }
    }
}