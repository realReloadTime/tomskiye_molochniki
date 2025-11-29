using Microsoft.AspNetCore.Mvc;
using BackendSite.Dtos;
using System.Text;

namespace BackendSite.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AnalysisController : ControllerBase
    {
        private readonly HttpClient _httpClient;

        public AnalysisController(IHttpClientFactory httpClientFactory)
        {
            _httpClient = httpClientFactory.CreateClient("PythonBackend");
        }

        [HttpPost("analyze")]
        public async Task<IActionResult> AnalyzeText([FromForm] string review)
        {
            if (string.IsNullOrEmpty(review))
                return BadRequest("Текст не может быть пустым");

            try
            {
                var formData = new MultipartFormDataContent();
                formData.Add(new StringContent(review), "text");

                var response = await _httpClient.PostAsync("/tonality/text", formData);

                if (response.IsSuccessStatusCode)
                {
                    var result = await response.Content.ReadFromJsonAsync<AnalysisResultDto>();
                    return Ok(result);
                }

                var errorContent = await response.Content.ReadAsStringAsync();
                return BadRequest($"Ошибка Python бэкенда: {response.StatusCode} - {errorContent}");
            }
            catch (Exception ex)
            {
                return BadRequest($"Ошибка соединения с Python бэкендом: {ex.Message}");
            }
        }

        [HttpPost("analyze-file")]
        public async Task<IActionResult> AnalyzeFile(IFormFile csvFile)
        {
            if (csvFile == null || csvFile.Length == 0)
                return BadRequest("Файл не выбран");

            try
            {
                var formData = new MultipartFormDataContent();

                using var fileStream = csvFile.OpenReadStream();
                var fileContent = new StreamContent(fileStream);
                formData.Add(fileContent, "file", csvFile.FileName);

                var response = await _httpClient.PostAsync("/tonality/file", formData);

                if (response.IsSuccessStatusCode)
                {
                    var csvContent = await response.Content.ReadAsStringAsync();

                    
                    Console.WriteLine($"Raw CSV response length: {csvContent.Length}");
                    Console.WriteLine($"First 500 chars: {csvContent.Substring(0, Math.Min(500, csvContent.Length))}");

                    var records = ParseCsvResponse(csvContent);

                    Console.WriteLine($"Parsed records: {records.Count}");
                    if (records.Count > 0)
                    {
                        Console.WriteLine($"First record: {records[0].Comment} | {records[0].ClassLabel} | {records[0].Probability}");
                    }

                    return Ok(new
                    {
                        message = "Файл успешно обработан",
                        fileName = csvFile.FileName,
                        records = records,
                        totalRecords = records.Count,
                        toxicCount = records.Count(r => r.ClassLabel == 1),
                        nonToxicCount = records.Count(r => r.ClassLabel == 0)
                    });
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    return BadRequest($"Ошибка Python бэкенда: {response.StatusCode} - {errorContent}");
                }
            }
            catch (Exception ex)
            {
                return BadRequest($"Ошибка обработки файла: {ex.Message}");
            }
        }

        
        private List<FileAnalysisRecord> ParseCsvResponse(string csvContent)
        {
            var records = new List<FileAnalysisRecord>();

            
            var lines = csvContent.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries);

            Console.WriteLine($"Total lines: {lines.Length}");

           
            for (int i = 1; i < lines.Length; i++)
            {
                var line = lines[i].Trim();
                if (string.IsNullOrEmpty(line)) continue;

                Console.WriteLine($"Processing line {i}: {line}");

                try
                {
                    
                    var firstComma = line.IndexOf(',');
                    if (firstComma == -1) continue;

                    var secondComma = line.IndexOf(',', firstComma + 1);
                    if (secondComma == -1) continue;

                    var comment = line.Substring(0, firstComma).Trim('"').Trim();
                    var classLabelStr = line.Substring(firstComma + 1, secondComma - firstComma - 1).Trim();
                    var probabilityStr = line.Substring(secondComma + 1).Trim();

                    
                    probabilityStr = probabilityStr.Replace("\r", "");

                    if (int.TryParse(classLabelStr, out int classLabel) &&
                        double.TryParse(probabilityStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double probability))
                    {
                        records.Add(new FileAnalysisRecord
                        {
                            Comment = comment,
                            ClassLabel = classLabel,
                            Probability = probability
                        });
                    }
                    else
                    {
                        Console.WriteLine($"Failed to parse: classLabel='{classLabelStr}', probability='{probabilityStr}'");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error parsing line {i}: {ex.Message}");
                }
            }

            return records;
        }
    }

    
    public class FileAnalysisRecord
    {
        public string Comment { get; set; } = string.Empty;
        public int ClassLabel { get; set; }
        public double Probability { get; set; }
    }
}