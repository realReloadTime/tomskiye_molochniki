using System.Text.Json.Serialization;

namespace BackendSite.Dtos
{
    public class AnalysisResultDto
    {
        [JsonPropertyName("comment")]
        public string comment { get; set; } = string.Empty;

        [JsonPropertyName("class_label")]
        public double class_label { get; set; }

        [JsonPropertyName("probability")]
        public double probability { get; set; }

        [JsonPropertyName("created_date")]
        public DateTime created_date { get; set; }
    }

    public class FileAnalysisResponseDto
    {
        public DateTime AnalysisDate { get; set; }
        public int TotalRecords { get; set; }
        public int PositiveCount { get; set; }
        public int NegativeCount { get; set; }
    }

    public class FileAnalysisRecordDto
    {
        public string Comment { get; set; } = string.Empty;
        public int ClassLabel { get; set; }
        public double Probability { get; set; }
    }

    public class FileAnalysisResult
    {
        public int TotalRecords { get; set; }
        public int PositiveCount { get; set; }
        public int NegativeCount { get; set; }
        public List<FileAnalysisRecordDto> Records { get; set; } = new();
    }
}