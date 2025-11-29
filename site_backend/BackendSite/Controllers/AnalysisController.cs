using Microsoft.AspNetCore.Mvc;
using BackendSite.Dtos;
using BackendSite.Services;

namespace BackendSite.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AnalysisController : ControllerBase
    {
        private readonly IAnalysisService _analysisService;

        public AnalysisController(IAnalysisService analysisService)
        {
            _analysisService = analysisService;
        }

        [HttpPost("analyze")]
        public async Task<IActionResult> AnalyzeText([FromForm] string review)
        {
            if (string.IsNullOrEmpty(review))
                return BadRequest("Текст не может быть пустым");

            var result = await _analysisService.AnalyzeTextAsync(review);

            if (result == null)
                return BadRequest("Ошибка при анализе текста");

            return Ok(result);
        }

        [HttpPost("analyze-file")]
        public async Task<IActionResult> AnalyzeFile(IFormFile csvFile)
        {
            if (csvFile == null || csvFile.Length == 0)
                return BadRequest("Файл не выбран");

            var result = await _analysisService.AnalyzeFileAsync(csvFile);

            if (result.TotalRecords == 0)
                return BadRequest("Не удалось обработать файл или файл пуст");

            return Ok(result);
        }
    }
}