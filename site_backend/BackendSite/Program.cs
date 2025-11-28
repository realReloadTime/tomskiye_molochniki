using Scalar.AspNetCore;
using Microsoft.EntityFrameworkCore;
using BackendSite.Models;
using BackendSite.Data;
using BackendSite.Services;
using BackendSite.Middleware;

var builder = WebApplication.CreateBuilder(args);


builder.Services.AddOpenApi();
builder.Services.AddControllers();


builder.Services.AddDbContext<ReviewsHackatonDBContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.AddScoped<IPasswordHasherService, PasswordHasherService>();
builder.Services.AddScoped<IAuthService, AuthService>();
builder.Services.AddScoped<ICookieService, CookieService>();
builder.Services.AddHttpContextAccessor(); 

var app = builder.Build();


app.UseRouting();
app.UseAuthorization();

app.MapControllers();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.MapScalarApiReference();
}

app.UseHttpsRedirection();
app.UseMiddleware<BackendSite.Middleware.SessionMiddleware>();

app.Run();