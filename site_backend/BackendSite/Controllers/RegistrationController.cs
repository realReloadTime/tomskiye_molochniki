using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using BackendSite.Data;
using BackendSite.Models;
using BackendSite.Dtos;
using BackendSite.Services;


namespace BackendSite.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class RegistrationController : ControllerBase
    {
        private readonly ReviewsHackatonDBContext _context;
        private readonly IPasswordHasherService _passwordHasher;

        public RegistrationController(
            ReviewsHackatonDBContext context,
            IPasswordHasherService passwordHasher)
        {
            _context = context;
            _passwordHasher = passwordHasher;
        }

        [HttpPost]
        public async Task<IActionResult> Register(UserRegistrationDto dto)
        {

            bool userExists = await _context.Users
                .AnyAsync(u => u.Login == dto.Login);

            if (userExists)
            {
                return BadRequest("Пользователь с таким логином уже существует");
            }

            if (dto.Password != dto.ConfirmPassword)
            {
                return BadRequest("Пароли не совпадают");
            }

            string passwordHash = _passwordHasher.HashPassword(dto.Password);

            var user = new User
            {
                Login = dto.Login.Trim(),
                PasswordHash = passwordHash
            };

            _context.Users.Add(user);
            await _context.SaveChangesAsync();

            return Ok("Пользователя зарегистрирован");
        }
    }
}