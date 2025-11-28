using BackendSite.Models;
using Microsoft.EntityFrameworkCore;

namespace BackendSite.Data
{

	public class ReviewsHackatonDBContext : DbContext
	{
		public ReviewsHackatonDBContext(DbContextOptions<ReviewsHackatonDBContext> options)
			: base(options)
		{
		}

		public DbSet<Statistic> Statistics { get; set; }
		public DbSet<User> Users { get; set; }
	}

}