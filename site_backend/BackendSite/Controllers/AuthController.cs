using Microsoft.AspNetCore.Mvc;
using BackendSite.Dtos;
using BackendSite.Services;

namespace BackendSite.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AuthController : ControllerBase
    {
        private readonly IAuthService _authService;
        private readonly ICookieService _cookieService;

        public AuthController(IAuthService authService, ICookieService cookieService)
        {
            _authService = authService;
            _cookieService = cookieService;
        }

        [HttpPost("login")]
        public async Task<ActionResult> Login(LoginDto dto)
        {
            var (user, error) = await _authService.AuthenticateAsync(dto.Login, dto.Password);

            if (user == null)
                return Unauthorized(error);

            var sessionToken = await _authService.CreateSessionAsync(user);
            _cookieService.SetSessionCookie(sessionToken);

            return Ok("Успешный вход");
        }

        [HttpPost("logout")]
        public async Task<ActionResult> Logout()
        {
            var sessionToken = _cookieService.GetSessionToken();

            if (!string.IsNullOrEmpty(sessionToken))
            {
                await _authService.LogoutAsync(sessionToken);
                _cookieService.DeleteSessionCookie();
            }

            return Ok("Выход выполнен успешно");
        }

        [HttpGet("check")]
        public async Task<ActionResult> CheckAuth()
        {
            var sessionToken = _cookieService.GetSessionToken();

            if (string.IsNullOrEmpty(sessionToken))
                return Unauthorized(new { isAuthenticated = false });

            var user = await _authService.GetUserFromSessionAsync(sessionToken);

            if (user == null)
                return Unauthorized(new { isAuthenticated = false });

            return Ok(new
            {
                isAuthenticated = true,
                userId = user.Id,
                login = user.Login
            });
        }
    }
}