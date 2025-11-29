namespace BackendSite.Dtos
{
    public class FileAnalysisRecord
    {
        public string Comment { get; set; } = string.Empty;
        public int ClassLabel { get; set; }
        public double Probability { get; set; }
    }
}