import UploadForm from '../components/UploadForm';

export default function Home() {
  return (
    <div className="page home">
      <h1>Анализ отзывов</h1>
      <p>Загрузите отзыв или CSV — сервер проанализирует тональность</p>
      <UploadForm />
    </div>
  );
}
