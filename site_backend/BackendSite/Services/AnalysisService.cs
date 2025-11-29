using BackendSite.Dtos;
using System.Globalization;
using System.Text;

namespace BackendSite.Services
{
    public interface IAnalysisService
    {
        Task<AnalysisResultDto?> AnalyzeTextAsync(string text);
        Task<FileAnalysisResponseDto> AnalyzeFileAsync(IFormFile csvFile);
    }

    public class AnalysisService : IAnalysisService
    {
        private readonly HttpClient _httpClient;

        public AnalysisService(IHttpClientFactory httpClientFactory)
        {
            _httpClient = httpClientFactory.CreateClient("PythonBackend");
        }

        public async Task<AnalysisResultDto?> AnalyzeTextAsync(string text)
        {
            try
            {
                var formData = new MultipartFormDataContent();
                formData.Add(new StringContent(text), "text");

                var response = await _httpClient.PostAsync("/tonality/text", formData);

                if (response.IsSuccessStatusCode)
                {
                    return await response.Content.ReadFromJsonAsync<AnalysisResultDto>();
                }
            }
            catch (Exception ex)
            {
    
                Console.WriteLine($"Error in AnalyzeTextAsync: {ex.Message}");
            }

            return null;
        }

        public async Task<FileAnalysisResponseDto> AnalyzeFileAsync(IFormFile csvFile)
        {
            try
            {
                var formData = new MultipartFormDataContent();

                using var fileStream = csvFile.OpenReadStream();
                var fileContent = new StreamContent(fileStream);
                formData.Add(fileContent, "file", csvFile.FileName);

                var response = await _httpClient.PostAsync("/tonality/file", formData);

                if (response.IsSuccessStatusCode)
                {
                    var csvBytes = await response.Content.ReadAsByteArrayAsync();
                    var csvContent = Encoding.UTF8.GetString(csvBytes);

                    var analysisResult = ParseCsvAnalysis(csvContent);

                    return new FileAnalysisResponseDto
                    {
                        AnalysisDate = DateTime.UtcNow,
                        TotalRecords = analysisResult.TotalRecords,
                        PositiveCount = analysisResult.PositiveCount,
                        NegativeCount = analysisResult.NegativeCount
                    };
                }
            }
            catch (Exception ex)
            {
          
                Console.WriteLine($"Error in AnalyzeFileAsync: {ex.Message}");
            }

            return new FileAnalysisResponseDto();
        }

        private FileAnalysisResult ParseCsvAnalysis(string csvContent)
        {
            var records = new List<FileAnalysisRecordDto>();

            if (csvContent.StartsWith("\"") && csvContent.EndsWith("\""))
            {
                csvContent = csvContent.Substring(1, csvContent.Length - 2);
            }

            var lines = csvContent.Split(new[] { "\\r\\n" }, StringSplitOptions.RemoveEmptyEntries);

            for (int i = 1; i < lines.Length; i++)
            {
                var line = lines[i].Trim();
                line = line.Replace("\\\"", "\"");

                if (string.IsNullOrEmpty(line)) continue;

                try
                {
                    var parts = ParseCsvLineSimple(line);
                    if (parts.Count >= 3)
                    {
                        var comment = parts[0].Trim('"').Trim();
                        if (int.TryParse(parts[1], out int classLabel) &&
                            double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out double probability))
                        {
                            records.Add(new FileAnalysisRecordDto
                            {
                                Comment = comment,
                                ClassLabel = classLabel,
                                Probability = probability
                            });
                        }
                    }
                }
                catch
                {
                    
                }
            }

            var total = records.Count;
            var positive = records.Count(r => r.ClassLabel == 0);
            var negative = records.Count(r => r.ClassLabel == 2);

            return new FileAnalysisResult
            {
                TotalRecords = total,
                PositiveCount = positive,
                NegativeCount = negative,
                Records = records
            };
        }

        private List<string> ParseCsvLineSimple(string line)
        {
            var result = new List<string>();
            var current = new StringBuilder();
            var inQuotes = false;

            for (int i = 0; i < line.Length; i++)
            {
                var ch = line[i];

                if (ch == '"')
                {
                    inQuotes = !inQuotes;
                    current.Append(ch);
                }
                else if (ch == ',' && !inQuotes)
                {
                    result.Add(current.ToString());
                    current.Clear();
                }
                else
                {
                    current.Append(ch);
                }
            }

            result.Add(current.ToString());
            return result;
        }
    }
}