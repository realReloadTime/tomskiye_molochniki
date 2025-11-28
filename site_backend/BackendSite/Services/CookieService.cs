using Microsoft.AspNetCore.Hosting;

namespace BackendSite.Services
{
    public class CookieService : ICookieService
    {
        private readonly IHttpContextAccessor _httpContextAccessor;
        private readonly IWebHostEnvironment _environment;

        public CookieService(IHttpContextAccessor httpContextAccessor, IWebHostEnvironment environment)
        {
            _httpContextAccessor = httpContextAccessor;
            _environment = environment;
        }

        public void SetSessionCookie(string sessionToken)
        {
            var options = new CookieOptions
            {
                HttpOnly = true,
                Secure = !_environment.IsDevelopment(),
                SameSite = SameSiteMode.Strict,
                Expires = DateTimeOffset.Now.AddDays(30),
                Path = "/"
            };

            _httpContextAccessor.HttpContext?.Response.Cookies.Append("session_token", sessionToken, options);
        }

        public string GetSessionToken() 
        {
            return _httpContextAccessor.HttpContext?.Request.Cookies["session_token"] ?? string.Empty;
        }

        public void DeleteSessionCookie()
        {
            _httpContextAccessor.HttpContext?.Response.Cookies.Delete("session_token");
        }
    }
}